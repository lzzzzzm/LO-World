#!/usr/bin/env python3
"""Generate LO-World 12 Hz occupancy predictions for a clip manifest."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config, DictAction
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[3]))

from tools.bevworld_generation.rollout import (
    build_layout_sequence,
    decode_occ,
    rollout,
)
from tools.bevworld_generation.dataset import (
    current_latent_and_memo_inputs,
    prepare_batch,
    warmup_memo,
)
from tools.bevworld_generation.points import (
    pack_point_sequence,
    points_to_numpy,
)
from tools.bevworld_generation.runtime import build_runner
from tools.genie.occ_generation.scene_runtime import sha256_file
from tools.genie.occ_generation.scene_timeline import set_decode_frame_metadata
from tools.trajectory_generation.bev_layout import get_lidar2ego
from tools.trajectory_generation.resample_scene_timeline import TIMELINE_VERSION


DEFAULT_CONFIG = "configs/world_model/bev_world_12f_layout_autoregressive.py"
DEFAULT_CHECKPOINT = (
    "ckpts/bev_world_12f_layout_autoregressive.pth")


def validate_condition_clock(path, future_frames):
    with np.load(path, allow_pickle=False) as data:
        if ('timeline_version' not in data or str(data['timeline_version']) != TIMELINE_VERSION
                or 'frame_timestamps_us' not in data):
            raise ValueError(f'Condition is missing the uniform 12 Hz clock: {path}')
        times = data['frame_timestamps_us']
        if times.shape != (future_frames+1,) or not np.array_equal(
                times, times[0]+np.rint(np.arange(future_frames+1)*1e6/12).astype(np.int64)):
            raise ValueError(f'Condition timestamps do not match uniform 12 Hz: {path}')
        token = str(data['target_token'])
        expected = [f'uniform12_{token}_{int(t)}' for t in times[1:]]
        if not np.array_equal(data['future_tokens'], expected):
            raise ValueError(f'Condition tokens do not match its physical timestamps: {path}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--dataset-dir", type=Path,
        default=Path("work_dirs/genie_dataset"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--warmup-frames", type=int, default=4)
    parser.add_argument(
        "--decode-chunk-size", type=int, default=13,
        help=("Frames decoded per standard obtain_data_from_latent call. "
              "Default 13 matches the config evaluation horizon."))
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--progress-dir", type=Path, default=None)
    parser.add_argument(
        "--finalize-only", action="store_true",
        help="Validate all prediction files and write the aggregate status.")
    parser.add_argument(
        "--dataset-split", choices=("train", "val", "test"), default="train",
        help="Config dataloader used to resolve manifest dataset indices.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--save-points", action="store_true",
        help=(
            "Jointly decode the network-predicted point cloud. Point frames "
            "are packed into each clip NPZ and also saved as one (N, 3) NPZ "
            "per future frame."))
    parser.add_argument(
        "--points-dir", type=Path, default=None,
        help="Per-frame point output root (default: <out-dir>/points_12hz).")
    parser.add_argument(
        "--cfg-options", nargs="+", action=DictAction,
        help="Override config options, same syntax as tools/test.py")
    return parser.parse_args()


def build_inference_dataset(args: argparse.Namespace):
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if cfg.get("custom_imports", None):
        from mmengine.utils import import_modules_from_strings
        import_modules_from_strings(**cfg.custom_imports)
    if cfg.get("default_scope", None):
        from mmengine.registry import init_default_scope
        init_default_scope(cfg.default_scope)
    dataloader_key = f"{args.dataset_split}_dataloader"
    if dataloader_key not in cfg:
        raise KeyError(f"{dataloader_key} is missing from {args.config}")
    dataset_cfg = cfg[dataloader_key].dataset.copy()
    dataset_cfg.test_mode = True
    dataset_cfg.load_futu_number = 0
    dataset_cfg.use_sequence_group_flag = True
    dataset_cfg.sequences_split_num = 1
    from mmdet3d.registry import DATASETS
    return DATASETS.build(dataset_cfg)


def valid_prediction(
        path: Path, token: str, frames: int,
        require_points: bool = False, condition_sha256: str | None = None,
        annotation_sha256: str | None = None) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            valid = (
                str(np.asarray(data["target_token"]).reshape(-1)[0]) == token
                and data["pred_occ_12hz"].shape == (frames, 200, 200, 16)
                and data["current_occ"].shape == (200, 200, 16)
                and (condition_sha256 is None or (
                    'condition_sha256' in data and str(data['condition_sha256']) == condition_sha256))
                and (annotation_sha256 is None or (
                    'selected_annotation_sha256' in data and str(data['selected_annotation_sha256']) == annotation_sha256))
            )
            if require_points:
                valid = valid and (
                    "pred_points_xyz_concat" in data
                    and "pred_points_xyz_offsets" in data
                    and data["pred_points_xyz_concat"].ndim == 2
                    and data["pred_points_xyz_concat"].shape[1] == 3
                    and data["pred_points_xyz_offsets"].shape == (frames + 1,)
                )
            return valid
    except Exception:
        return False


@torch.no_grad()
def decode_occ_and_points(
        model, latents, batch_data_samples, chunk_size: int,
        progress_desc: str | None = None, start_frame: int | None = None):
    """Jointly decode OCC/LiDAR, optionally indexing a complete frame timeline.

    A supplied start_frame advances with each latent chunk. None preserves the
    existing API for callers whose metadata is local to each decode call.
    """
    if chunk_size <= 0:
        raise ValueError('decode chunk_size must be positive')
    occ_frames = []
    point_frames = []
    starts = range(0, latents.size(1), chunk_size)
    if progress_desc is not None:
        starts = tqdm(starts, total=(latents.size(1) + chunk_size - 1) // chunk_size,
                      desc=progress_desc)
    for start in starts:
        chunk = latents[:, start:start + chunk_size]
        flat = chunk.reshape(-1, *chunk.shape[2:])
        metadata_options = {} if start_frame is None else dict(start_frame=start_frame + start)
        occ, points = model.obtain_data_from_latent(
            flat,
            batch_data_samples=batch_data_samples,
            frames_per_sample=chunk.size(1), **metadata_options)
        occ_frames.append(occ.to(torch.uint8).cpu().numpy())
        point_frames.extend(points_to_numpy(points) for points in points)
        del occ, points, flat, chunk
    return np.concatenate(occ_frames, axis=0), point_frames


def save_point_frames(
        points_root: Path, clip_index: int, target_token: str,
        future_tokens: np.ndarray, points_sequence,
        coordinate: str = "lidar") -> None:
    """Save one pickle-free NPZ containing an (N, 3) array per frame."""
    clip_dir = points_root / f"{clip_index:06d}"
    clip_dir.mkdir(parents=True, exist_ok=True)
    if len(points_sequence) != len(future_tokens):
        raise ValueError(
            f"Point/token frame mismatch: {len(points_sequence)} versus "
            f"{len(future_tokens)}")
    for frame_index, (future_token, points) in enumerate(
            zip(future_tokens, points_sequence)):
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                f"Expected point frame (N, 3), got {points.shape}")
        output_path = clip_dir / f"{frame_index:06d}.npz"
        temporary_path = output_path.with_suffix(".npz.tmp")
        with temporary_path.open("wb") as file:
            np.savez_compressed(
                file,
                points=points,
                frame_index=np.asarray(frame_index, dtype=np.int64),
                target_token=np.asarray(target_token),
                future_token=np.asarray(str(future_token)),
                coordinate=np.asarray(coordinate),
            )
        os.replace(temporary_path, output_path)


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("--shard-id must be in [0, --num-shards)")
    manifest_path = args.manifest or args.dataset_dir / "selection_manifest.json"
    out_dir = args.out_dir or args.dataset_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    records = manifest["records"]
    future_frames = int(manifest["future_frames"])
    if manifest.get('timeline_version') != TIMELINE_VERSION:
        raise ValueError('Dataset uses an unsupported old timeline. Rebuild with prepare_dataset.py.')
    selected_annotation = args.dataset_dir / manifest.get('compact_annotation', 'selected_12hz_frames.pkl')
    annotation_hash = sha256_file(selected_annotation)
    if manifest.get('compact_annotation_sha256', annotation_hash) != annotation_hash:
        raise ValueError('Selected annotation changed after dataset preparation')
    condition_hashes = {}
    for record in records:
        condition_path = args.dataset_dir / record['condition_file']
        validate_condition_clock(condition_path, future_frames)
        actual = sha256_file(condition_path)
        if record.get('condition_sha256', actual) != actual:
            raise ValueError(f'Condition changed after dataset preparation: {condition_path}')
        condition_hashes[int(record['clip_index'])] = actual

    if args.finalize_only:
        missing = []
        for record in records:
            path = out_dir / f"{int(record['clip_index']):06d}.npz"
            if not valid_prediction(
                    path, str(record["target_token"]), future_frames,
                    require_points=args.save_points,
                    condition_sha256=condition_hashes[int(record['clip_index'])], annotation_sha256=annotation_hash):
                missing.append(int(record["clip_index"]))
        if missing:
            raise RuntimeError(
                f"Cannot finalize: {len(missing)} invalid/missing predictions; "
                f"first={missing[:10]}")
        status_path = out_dir / "generation_status.json"
        with status_path.open("w", encoding="utf-8") as file:
            json.dump(dict(
                completed=len(records), config=args.config,
                checkpoint=args.checkpoint, num_shards=args.num_shards),
                file, indent=2, ensure_ascii=False)
        print(f"Validated all {len(records)} prediction clips in {out_dir}")
        return

    assigned = [
        record for record in records
        if int(record["clip_index"]) % args.num_shards == args.shard_id
    ]

    pending = []
    for record in assigned:
        path = out_dir / f"{int(record['clip_index']):06d}.npz"
        if args.overwrite or not valid_prediction(
                path, str(record["target_token"]), future_frames,
                require_points=args.save_points,
                condition_sha256=condition_hashes[int(record['clip_index'])], annotation_sha256=annotation_hash):
            pending.append(record)
        elif args.progress_dir is not None:
            args.progress_dir.mkdir(parents=True, exist_ok=True)
            (args.progress_dir / f"reused_{int(record['clip_index']):06d}").touch()
    if not pending:
        print(
            f"Shard {args.shard_id}/{args.num_shards}: all "
            f"{len(assigned)} assigned clips are already complete")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("LO-World batch generation requires a CUDA GPU")

    decode_frames = {}
    if args.save_points:
        with selected_annotation.open('rb') as stream:
            compact = pickle.load(stream)
        decode_frames = {int(clip['record']['clip_index']): clip['frames']
                         for clip in compact['clips']}

    print(
        f"Shard {args.shard_id}/{args.num_shards}: loading LO-World once "
        f"for {len(pending)} pending clips", flush=True)
    runner = build_runner(args)
    model = runner.model
    model.eval()
    dataset = build_inference_dataset(args)

    progress = tqdm(
        pending, desc=f"LO-World 12 Hz shard {args.shard_id}")
    runtime_rows = []
    for record in progress:
        clip_index = int(record["clip_index"])
        token = str(record["target_token"])
        dataset_index = int(record["lo_dataset_index"])
        info = dataset.get_data_info(dataset_index)
        if str(info["token"]) != token:
            raise ValueError(
                f"Dataset index {dataset_index} token {info['token']} != {token}")
        condition_path = args.dataset_dir / record["condition_file"]
        with np.load(condition_path, allow_pickle=False) as condition:
            layout_np = np.asarray(condition["bev_layout"][0], dtype=np.uint8)
            ego_np = np.asarray(
                condition["candidate_curr2futu_ego"][0], dtype=np.float32)
            future_tokens = np.asarray(condition["future_tokens"]).astype(str)
            timeline_version = str(condition['timeline_version'])
            if timeline_version != TIMELINE_VERSION:
                raise ValueError(f'Condition uses an unsupported old timeline: {condition_path}')
            frame_timestamps = condition['frame_timestamps_us'].copy()

        warmed_tokens = warmup_memo(
            model, dataset, dataset_index, args.warmup_frames)
        data = prepare_batch(model, dataset, dataset_index)
        batch_inputs, batch_data_samples, current_latent, start_sequence = (
            current_latent_and_memo_inputs(model, data))
        memo_latent = model.get_memo_latent(current_latent, start_sequence)
        layout = build_layout_sequence(
            model, batch_inputs["bev_layout"], layout_np)
        ego = torch.as_tensor(
            ego_np, device=current_latent.device,
            dtype=torch.float32).unsqueeze(0)
        seed = int(record["generation_seed"])
        future_latent, seconds, peak_mb = rollout(
            model, current_latent, memo_latent, ego, layout,
            future_frames, seed)
        if args.save_points:
            set_decode_frame_metadata(batch_data_samples, decode_frames[clip_index], model)
            all_latents = torch.cat([current_latent, future_latent], dim=1)
            all_occ, all_points = decode_occ_and_points(
                model, all_latents, batch_data_samples,
                args.decode_chunk_size, start_frame=model.prev_number)
            current_occ = all_occ[0]
            future_occ = all_occ[1:]
            current_points = np.asarray(
                all_points[0][:, :3], dtype=np.float32)
            future_points = [
                np.asarray(points[:, :3], dtype=np.float32)
                for points in all_points[1:]
            ]
        else:
            future_occ = decode_occ(
                model, future_latent, args.decode_chunk_size)
            future_points = None
            current_points = None
        if not args.save_points:
            current_occ = decode_occ(model, current_latent, 1)[0]
        output_path = out_dir / f"{clip_index:06d}.npz"
        temporary_path = output_path.with_suffix(".npz.tmp")
        save_dict = dict(
            target_token=np.asarray(token),
            future_tokens_12hz=future_tokens,
            current_occ=current_occ,
            pred_occ_12hz=future_occ,
            condition_file=np.asarray(str(condition_path.resolve())),
            generation_seed=np.asarray(seed, dtype=np.int64),
            condition_sha256=np.asarray(condition_hashes[clip_index]),
            timeline_version=np.asarray(timeline_version),
            selected_annotation_sha256=np.asarray(annotation_hash),
        )
        if frame_timestamps is not None:
            save_dict['frame_timestamps_us'] = frame_timestamps
        if future_points is not None:
            packed = pack_point_sequence(future_points, default_dim=3)
            sample_meta = dict(batch_data_samples[0].metainfo)
            lidar2ego = sample_meta.get("lidar2ego")
            if lidar2ego is None:
                total_lidar2ego = sample_meta.get("total_lidar2ego")
                if total_lidar2ego is not None:
                    lidar2ego = np.asarray(total_lidar2ego)[model.prev_number]
            save_dict.update(
                point_coordinate=np.asarray("lidar"),
                current_points_xyz=current_points,
                pred_points_xyz_concat=packed["concat"],
                pred_points_xyz_offsets=packed["offsets"],
                pred_points_xyz_dim=packed["dim"],
            )
            if lidar2ego is not None:
                save_dict["point_lidar2ego"] = np.asarray(
                    lidar2ego, dtype=np.float32)
            save_dict['pred_points_lidar2ego'] = np.asarray(
                [get_lidar2ego(frame) for frame in decode_frames[clip_index][1:]],
                dtype=np.float32)
        with temporary_path.open("wb") as file:
            np.savez_compressed(file, **save_dict)
        os.replace(temporary_path, output_path)
        if args.progress_dir is not None:
            args.progress_dir.mkdir(parents=True, exist_ok=True)
            (args.progress_dir / f"generated_{clip_index:06d}").touch()
        if future_points is not None:
            points_root = args.points_dir or (out_dir / "points_12hz")
            save_point_frames(
                points_root, clip_index, token, future_tokens,
                future_points)
        runtime_rows.append(dict(
            clip_index=clip_index, target_token=token,
            warmed_tokens=[str(x) for x in warmed_tokens],
            seconds=seconds, peak_allocated_mib=peak_mb))
        progress.set_postfix(seconds=f"{seconds:.2f}", peak_mib=f"{peak_mb:.0f}")

    status_path = out_dir / (
        "generation_status.json" if args.num_shards == 1
        else f"generation_status_shard{args.shard_id:02d}.json")
    with status_path.open("w", encoding="utf-8") as file:
        json.dump(dict(
            shard_id=args.shard_id, num_shards=args.num_shards,
            assigned=len(assigned), generated_this_run=len(runtime_rows),
            config=args.config, checkpoint=args.checkpoint,
            runtime=runtime_rows), file, indent=2, ensure_ascii=False)
    print(
        f"Shard {args.shard_id}/{args.num_shards} completed "
        f"{len(assigned)} clips in {out_dir}")


if __name__ == "__main__":
    main()
