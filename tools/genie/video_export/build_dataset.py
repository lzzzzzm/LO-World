#!/usr/bin/env python3
"""Render a manifest of predicted OCC clips into Genie six-camera controls."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[3]))

from tools.genie.semantic_export.renderer import (
    CAM_NAMES,
    create_full_center_coords,
    load_prediction_clip,
    make_writer,
    render_occ_frame,
    resolve_image_path,
    validate_future_tokens,
)
from tools.genie.occ_generation.scene_runtime import identity_key, sha256_file
from tools.genie.semantic_export.scene_inputs import validate_prediction_clock
from tools.trajectory_generation.resample_scene_timeline import TIMELINE_VERSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", type=Path,
        default=Path("work_dirs/genie_dataset"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--selected-ann", type=Path, default=None)
    parser.add_argument("--predictions-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--data-root", type=Path,
        default=Path("data/nuscenes"))
    parser.add_argument("--fps", type=int, choices=(12,), default=12)
    parser.add_argument("--height", type=int, default=450)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--gaussian-scale", type=float, default=0.2)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--progress-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    return parser.parse_args()


def output_paths(out_dir: Path, clip_index: int, prefix: str) -> dict[str, Path]:
    return {
        cam: out_dir / f"{prefix}_{clip_index:06d}_{cam}.mp4"
        for cam in CAM_NAMES
    }


def clip_manifest_path(out_dir: Path, clip_index: int) -> Path:
    return out_dir / "clip_manifests" / f"{clip_index:06d}.json"


def is_complete(
    out_dir: Path,
    clip_index: int,
    frames: int,
    height: int | None = None,
    width: int | None = None,
    fps: int | None = None,
    source_signature: str | None = None,
) -> bool:
    manifest_path = clip_manifest_path(out_dir, clip_index)
    if not manifest_path.is_file():
        return False
    try:
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        if int(manifest["frames"]) != frames:
            return False
        if height is not None and manifest.get(
                "condition_resolution") != [height, width]:
            return False
        if fps is not None and int(manifest.get("fps", -1)) != fps:
            return False
        if source_signature is not None and manifest.get('source_signature') != source_signature:
            return False
        for key in ("rgb_paths", "control_paths"):
            if set(manifest[key]) != set(CAM_NAMES):
                return False
            for relpath in manifest[key].values():
                path = out_dir / relpath
                if not path.is_file() or path.stat().st_size == 0:
                    return False
        return True
    except Exception:
        return False


def finalize(out_dir: Path, records: list[dict], future_frames: int) -> None:
    meta = []
    missing = []
    for record in sorted(records, key=lambda row: int(row["clip_index"])):
        clip_index = int(record["clip_index"])
        if not is_complete(out_dir, clip_index, future_frames + 1,
                           source_signature=record.get('_export_signature')):
            missing.append(clip_index)
            continue
        with clip_manifest_path(out_dir, clip_index).open(
                "r", encoding="utf-8") as file:
            clip = json.load(file)
        row = {"text": "", "type": "video"}
        for cam in CAM_NAMES:
            row[f"file_path_{cam}"] = clip["rgb_paths"][cam]
            row[f"control_file_path_{cam}"] = clip["control_paths"][cam]
        row.update(
            _customavwm_scene_name=record["interp_scene_name"],
            _customavwm_token=record["target_token"],
            _condition_source=(
                "LO-World 12 Hz predicted OCC via official GenieDrive GS"),
            _clip_index=clip_index,
            _generation_seed=int(record["generation_seed"]),
        )
        meta.append(row)
    if missing:
        raise RuntimeError(
            f"Cannot finalize: {len(missing)} incomplete clips, first={missing[:10]}")
    temporary = out_dir / "meta.json.tmp"
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2, ensure_ascii=False)
    os.replace(temporary, out_dir / "meta.json")
    print(f"Finalized {len(meta)} training records in {out_dir / 'meta.json'}")


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("--shard-id must be in [0, --num-shards)")
    manifest_path = args.manifest or args.dataset_dir / "selection_manifest.json"
    selected_ann = args.selected_ann or args.dataset_dir / "selected_12hz_frames.pkl"
    predictions_dir = args.predictions_dir or args.dataset_dir / "predictions"
    out_dir = args.out_dir or args.dataset_dir / "genie_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "clip_manifests").mkdir(parents=True, exist_ok=True)
    with manifest_path.open("r", encoding="utf-8") as file:
        selection = json.load(file)
    records = selection["records"]
    future_frames = int(selection["future_frames"])
    annotation_hash = sha256_file(selected_ann)
    if selection['compact_annotation_sha256'] != annotation_hash:
        raise ValueError('Selected annotation differs from its dataset manifest')
    if selection['timeline_version'] != TIMELINE_VERSION:
        raise ValueError('Dataset must use the uniform physical 12 Hz timeline')
    if args.fps != int(selection['fps']):
        raise ValueError('Export FPS must match the uniform physical timeline')
    for record in records:
        prediction = predictions_dir / f"{int(record['clip_index']):06d}.npz"
        record['_export_signature'] = identity_key(dict(
            prediction_sha256=sha256_file(prediction) if prediction.is_file() else 'missing',
            annotation_sha256=annotation_hash, height=args.height, width=args.width,
            fps=args.fps, gaussian_scale=args.gaussian_scale))
    if args.finalize_only:
        finalize(out_dir, records, future_frames)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Gaussian rendering requires a CUDA GPU")

    with selected_ann.open("rb") as file:
        compact = pickle.load(file)
    frames_by_index = {
        int(item["record"]["clip_index"]): item["frames"]
        for item in compact["clips"]
    }
    record_by_index = {int(row["clip_index"]): row for row in records}
    indices = []
    for index in sorted(record_by_index):
        if index % args.num_shards != args.shard_id:
            continue
        complete = is_complete(
            out_dir, index, future_frames + 1,
            args.height, args.width, args.fps, record_by_index[index]['_export_signature'])
        if args.overwrite or not complete:
            indices.append(index)
        elif args.progress_dir is not None:
            args.progress_dir.mkdir(parents=True, exist_ok=True)
            (args.progress_dir / f"reused_{index:06d}").touch()
    if not indices:
        print(f"Shard {args.shard_id}/{args.num_shards}: nothing pending")
        return

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    from third_party.geniedrive_occ_rasterizer.gaussian_renderer import render

    xyz = create_full_center_coords(device)
    rot = torch.zeros((xyz.shape[0], 4), device=device)
    rot[:, 0] = 1.0
    scale = torch.full(
        (xyz.shape[0], 3), args.gaussian_scale,
        device=device, dtype=torch.float32)

    for clip_index in tqdm(indices, desc=f"GS shard {args.shard_id}"):
        record = record_by_index[clip_index]
        frames = frames_by_index[clip_index]
        token, occ_frames, future_tokens = load_prediction_clip(
            predictions_dir / f"{clip_index:06d}.npz", future_frames)
        if token != str(record["target_token"]):
            raise ValueError(f"Clip {clip_index}: prediction token mismatch")
        validate_future_tokens(future_tokens, frames)
        prediction_path = predictions_dir / f'{clip_index:06d}.npz'
        validate_prediction_clock(prediction_path, frames, args.fps)
        with np.load(prediction_path, allow_pickle=False) as archive:
            if str(archive['selected_annotation_sha256']) != annotation_hash:
                raise ValueError('Prediction was generated with a different selected annotation')
        rgb_paths = output_paths(out_dir, clip_index, "RGB")
        control_paths = output_paths(out_dir, clip_index, "SEMANTIC")
        rgb_writers = {}
        control_writers = {}
        try:
            for cam in CAM_NAMES:
                control_writers[cam] = make_writer(
                    control_paths[cam], (args.width, args.height), args.fps)
            for occ, frame_info in zip(occ_frames, frames):
                colors, _ = render_occ_frame(
                    render, occ, frame_info, xyz, rot, scale, device,
                    (args.height, args.width))
                for cam in CAM_NAMES:
                    image_info = frame_info["images"][cam]
                    image_path = resolve_image_path(
                        image_info['img_path'],
                        args.data_root)
                    rgb = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                    if rgb is None:
                        raise RuntimeError(f"Failed to read {image_path}")
                    if cam not in rgb_writers:
                        rgb_writers[cam] = make_writer(
                            rgb_paths[cam], (rgb.shape[1], rgb.shape[0]), args.fps)
                    rgb_writers[cam].write(rgb)
                    control_writers[cam].write(colors[cam])
        finally:
            for writer in rgb_writers.values():
                writer.release()
            for writer in control_writers.values():
                writer.release()

        clip_manifest = dict(
            source_signature=record['_export_signature'],
            prediction_sha256=sha256_file(prediction_path),
            annotation_sha256=annotation_hash,
            timeline_version=record.get('timeline_version', 'legacy'),
            clip_index=clip_index,
            target_token=token,
            scene_name=record["interp_scene_name"],
            frames=len(frames),
            fps=args.fps,
            condition_resolution=[args.height, args.width],
            rgb_paths={cam: rgb_paths[cam].name for cam in CAM_NAMES},
            control_paths={cam: control_paths[cam].name for cam in CAM_NAMES},
            predictions=str((predictions_dir / f"{clip_index:06d}.npz").resolve()),
            gaussian_scale=args.gaussian_scale,
        )
        manifest_output = clip_manifest_path(out_dir, clip_index)
        temporary = manifest_output.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(clip_manifest, file, indent=2, ensure_ascii=False)
        os.replace(temporary, manifest_output)
        if args.progress_dir is not None:
            args.progress_dir.mkdir(parents=True, exist_ok=True)
            (args.progress_dir / f"generated_{clip_index:06d}").touch()
    print(f"Shard {args.shard_id}/{args.num_shards} completed {len(indices)} clips")


if __name__ == "__main__":
    main()
