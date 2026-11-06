#!/usr/bin/env python3
"""Select split-specific clips and cache their 12 Hz ground-truth BEV conditions.

The selected current frames are native nuScenes keyframes understood by
LO-World. Released annotations define clip selection; future geometry is
resampled from raw poses and instance tracks onto physical 12 Hz timestamps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[3]))

from tools.trajectory_generation.bev_layout import (
    current_to_future_transforms, render_layout_sequence,
    BEVLayoutRenderer,
    get_ego2global,
    get_info_location,
    get_lidar2ego,
    keyframe_info_to_render_info,
    load_infos,
    load_scene_locations,
)
from tools.genie.occ_generation.scene_runtime import identity_key, sha256_file, file_identity, atomic_file
from tools.trajectory_generation.resample_scene_timeline import (
    DEFAULT_TIMELINE_CACHE, TIMELINE_VERSION, resample_frames, validate_timeline,
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lo-ann-file", type=Path,
        default=Path("data/nus_infos/nuscenes_infos_train.pkl"))
    parser.add_argument(
        "--ann-12hz", type=Path,
        default=Path(
            "data/nus_infos_12hz/"
            "nuscenes_interp_12Hz_infos_train.pkl"))
    parser.add_argument("--data-root", type=Path, default=Path("data/nuscenes"))
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("work_dirs/genie_dataset"))
    parser.add_argument(
        "--split", choices=("train", "val"), default="train",
        help="Dataset split recorded in the output manifest.")
    parser.add_argument(
        "--exclude-locations", nargs="*", default=(),
        help="Skip map locations with pathological online rasterization cost.")
    parser.add_argument(
        "--allow-scene-repeats", action="store_true",
        help=(
            "If scene-distinct clips are insufficient, fill the requested "
            "count with unique start tokens from already selected scenes."))
    parser.add_argument("--num-clips", type=int, default=700)
    parser.add_argument(
        "--target-tokens", nargs="*", default=None,
        help=(
            "Explicit native keyframe tokens to package, in the requested "
            "order. This bypasses diverse random selection."))
    parser.add_argument(
        "--target-tokens-file", type=Path, default=None,
        help=(
            "JSON file containing a token list, or a dictionary with a "
            "target_tokens list. Useful for benchmark-defined clip order."))
    parser.add_argument("--future-frames", type=int, default=80)
    parser.add_argument("--warmup-frames", type=int, default=4)
    parser.add_argument("--turn-threshold-deg", type=float, default=10.0)
    parser.add_argument("--low-motion-threshold-m", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Rebuild preparation outputs instead of validating and reusing them.")
    parser.add_argument('--timeline-cache-dir', type=Path, default=DEFAULT_TIMELINE_CACHE)
    parser.add_argument('--map-geometry-cache', choices=('off', 'spatial'), default='off',
                        help='Optional exact spatial culling/cache before SDK map rasterization')
    return parser.parse_args(argv)


def yaw_from_pose(pose: np.ndarray) -> float:
    return float(np.arctan2(pose[1, 0], pose[0, 0]))


def motion_metrics(current: dict, future: dict) -> tuple[float, float]:
    current_pose = get_ego2global(current)
    future_pose = get_ego2global(future)
    relative = np.linalg.inv(current_pose) @ future_pose
    yaw_deg = float(np.degrees(yaw_from_pose(relative)))
    distance_m = float(np.linalg.norm(relative[:2, 3]))
    return yaw_deg, distance_m


def has_renderable_instances(info: dict) -> bool:
    """Return whether LO-World's dynamic-layout renderer can consume a frame."""
    if "gt_boxes" in info and "gt_names" in info:
        return True
    if "instances" not in info:
        return False
    return all(
        "bbox_3d" in instance
        for instance in info["instances"]
        if bool(instance.get("bbox_3d_isvalid", True))
    )


def eligible_candidates(
    lo_infos: list[dict], infos12: list[dict], future_frames: int,
    warmup_frames: int,
) -> list[dict]:
    index12 = {str(info["token"]): index for index, info in enumerate(infos12)}
    lo_groups: dict[str, list[int]] = defaultdict(list)
    for index, info in enumerate(lo_infos):
        lo_groups[str(info["scene_name"])].append(index)

    candidates = []
    native_steps = future_frames // 6
    for lo_scene, indices in tqdm(lo_groups.items(), desc="Match split tokens"):
        for local_index in range(warmup_frames, len(indices) - native_steps):
            lo_index = indices[local_index]
            token = str(lo_infos[lo_index]["token"])
            interp_index = index12.get(token)
            if interp_index is None or interp_index + future_frames >= len(infos12):
                continue
            frames = infos12[interp_index:interp_index + future_frames + 1]
            interp_scene = str(frames[0]["scene_name"])
            if any(str(frame["scene_name"]) != interp_scene for frame in frames):
                continue
            # A small number of interpolated train frames are missing the
            # v2-format instance list.  Reject only windows containing those
            # frames so every selected scene can fall back to another start.
            if any(not has_renderable_instances(frame) for frame in frames[1:]):
                continue
            expected_native = [
                str(lo_infos[indices[local_index + step]]["token"])
                for step in range(1, native_steps + 1)
            ]
            actual_native = [str(frames[step * 6]["token"])
                             for step in range(1, native_steps + 1)]
            if expected_native != actual_native:
                continue
            yaw_deg, distance_m = motion_metrics(frames[0], frames[-1])
            candidates.append(dict(
                lo_scene_name=lo_scene,
                interp_scene_name=interp_scene,
                location=str(frames[0].get("location", "")),
                lo_dataset_index=lo_index,
                lo_scene_local_index=local_index,
                interp_index=interp_index,
                target_token=token,
                final_yaw_deg=yaw_deg,
                displacement_m=distance_m,
            ))
    return candidates


def explicit_candidates(
    lo_infos: list[dict], infos12: list[dict], target_tokens: list[str],
    future_frames: int,
) -> list[dict]:
    """Resolve benchmark tokens without imposing random-selection warm-up.

    Benchmark protocols may begin at the first keyframe of each scene. Runtime
    memo warm-up already handles that case by using every available preceding
    frame (zero at a scene boundary), so explicit targets only need a valid
    contiguous future window.
    """
    lo_index = {str(info["token"]): index for index, info in enumerate(lo_infos)}
    interp_index = {
        str(info["token"]): index for index, info in enumerate(infos12)}
    lo_scene_local = {}
    scene_counts = defaultdict(int)
    for index, info in enumerate(lo_infos):
        scene = str(info["scene_name"])
        lo_scene_local[index] = scene_counts[scene]
        scene_counts[scene] += 1

    selected = []
    missing = []
    for token in target_tokens:
        li = lo_index.get(token)
        ii = interp_index.get(token)
        if li is None or ii is None:
            missing.append(token)
            continue
        frames = infos12[ii:ii + future_frames + 1]
        if len(frames) != future_frames + 1:
            missing.append(token)
            continue
        scene = str(frames[0]["scene_name"])
        if any(str(frame["scene_name"]) != scene for frame in frames):
            missing.append(token)
            continue
        yaw_deg, distance_m = motion_metrics(frames[0], frames[-1])
        missing_dynamic_offsets = [
            offset for offset, frame in enumerate(frames[1:], start=1)
            if not has_renderable_instances(frame)
        ]
        selected.append(dict(
            lo_scene_name=str(lo_infos[li]["scene_name"]),
            interp_scene_name=scene,
            location=str(frames[0].get("location", "")),
            lo_dataset_index=li,
            lo_scene_local_index=lo_scene_local[li],
            interp_index=ii,
            target_token=token,
            final_yaw_deg=yaw_deg,
            displacement_m=distance_m,
            motion_category="explicit_target",
            missing_dynamic_offsets=missing_dynamic_offsets,
        ))
    if missing:
        raise ValueError(
            f"Explicit target tokens lack a valid future window: {missing}")
    return selected


def valid_cached_condition(
    path: Path, target_token: str, future_tokens: list[str], future_frames: int,
    timeline_signature: str | None = None,
) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            cached_future = np.asarray(data["future_tokens"]).astype(str)
            return (
                str(np.asarray(data["target_token"]).reshape(-1)[0])
                == target_token
                and cached_future.tolist() == future_tokens
                and data["bev_layout"].shape
                == (1, future_frames, 18, 200, 200)
                and data["candidate_curr2futu_ego"].shape
                == (1, future_frames, 4, 4)
                and (timeline_signature is None or (
                    'timeline_signature' in data and str(data['timeline_signature']) == timeline_signature))
            )
    except Exception:
        return False


def select_diverse(
    candidates: list[dict], count: int, turn_threshold: float,
    low_motion_threshold: float, seed: int,
    allow_scene_repeats: bool = False,
) -> list[dict]:
    rng = random.Random(seed)
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for candidate in candidates:
        by_scene[candidate["lo_scene_name"]].append(candidate)

    scene_options = []
    for scene_name, rows in by_scene.items():
        left = max(rows, key=lambda row: row["final_yaw_deg"])
        right = min(rows, key=lambda row: row["final_yaw_deg"])
        low = min(rows, key=lambda row: row["displacement_m"])
        straight_rows = [
            row for row in rows
            if abs(row["final_yaw_deg"]) <= turn_threshold
            and row["displacement_m"] >= low_motion_threshold
        ]
        straight = rng.choice(straight_rows) if straight_rows else None
        scene_options.append(dict(
            scene_name=scene_name, left=left, right=right, low=low,
            straight=straight))

    target = {
        "left": round(count * 0.15),
        "right": round(count * 0.15),
        "low": round(count * 0.10),
    }
    target["straight"] = count - sum(target.values())
    pools = {
        "left": [x for x in scene_options
                 if x["left"]["final_yaw_deg"] >= turn_threshold],
        "right": [x for x in scene_options
                  if x["right"]["final_yaw_deg"] <= -turn_threshold],
        "low": [x for x in scene_options
                if x["low"]["displacement_m"] < low_motion_threshold],
        "straight": [x for x in scene_options if x["straight"] is not None],
    }
    for pool in pools.values():
        rng.shuffle(pool)

    selected = []
    used_scenes = set()
    for category in ("left", "right", "low", "straight"):
        for option in pools[category]:
            if option["scene_name"] in used_scenes:
                continue
            row = dict(option[category])
            row["motion_category"] = category
            selected.append(row)
            used_scenes.add(option["scene_name"])
            if sum(x["motion_category"] == category for x in selected) >= target[category]:
                break

    if len(selected) < count:
        fallback = candidates.copy()
        rng.shuffle(fallback)
        for candidate in fallback:
            if candidate["lo_scene_name"] in used_scenes:
                continue
            row = dict(candidate)
            row["motion_category"] = "fallback"
            selected.append(row)
            used_scenes.add(candidate["lo_scene_name"])
            if len(selected) == count:
                break
    if len(selected) < count and allow_scene_repeats:
        selected_tokens = {row["target_token"] for row in selected}
        repeated_fallback = candidates.copy()
        rng.shuffle(repeated_fallback)
        for candidate in repeated_fallback:
            if candidate["target_token"] in selected_tokens:
                continue
            row = dict(candidate)
            row["motion_category"] = "fallback_repeat_scene"
            selected.append(row)
            selected_tokens.add(row["target_token"])
            if len(selected) == count:
                break
    if len(selected) < count:
        raise RuntimeError(
            f"Only selected {len(selected)} scene-distinct clips; requested {count}")
    rng.shuffle(selected)
    return selected[:count]


def requested_target_tokens(args):
    if args.target_tokens is not None and args.target_tokens_file is not None:
        raise ValueError(
            "Use only one of --target-tokens and --target-tokens-file")
    target_tokens = args.target_tokens
    if args.target_tokens_file is not None:
        payload = json.loads(
            args.target_tokens_file.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if "target_tokens" in payload:
                target_tokens = payload["target_tokens"]
            elif "records" in payload:
                target_tokens = [
                    record["target_token"] for record in payload["records"]
                ]
            else:
                raise KeyError(
                    "Target-token JSON must contain 'target_tokens' or 'records'")
        else:
            target_tokens = payload
        if not isinstance(target_tokens, list):
            raise TypeError("Target-token JSON must contain a list")
        target_tokens = [str(token) for token in target_tokens]
    return target_tokens


def preparation_request(args, target_tokens):
    """Record the inputs and options that determine selection and conditions."""
    return dict(
        split=args.split, num_clips=len(target_tokens) if target_tokens else args.num_clips,
        future_frames=args.future_frames, warmup_frames=args.warmup_frames,
        seed=args.seed, turn_threshold_deg=args.turn_threshold_deg,
        low_motion_threshold_m=args.low_motion_threshold_m,
        excluded_locations=sorted(set(args.exclude_locations)),
        allow_scene_repeats=args.allow_scene_repeats, target_tokens=target_tokens,
        data_root=str(args.data_root.resolve()), version=args.version,
        map_geometry_cache=args.map_geometry_cache,
        lo_annotation=file_identity(args.lo_ann_file),
        annotation_12hz=file_identity(args.ann_12hz),
        geometry_sources=[file_identity(path) for path in (
            sorted((args.data_root / args.version).glob('*.json')) +
            sorted((args.data_root / 'maps/expansion').glob('*.json')))],
    )


def validate_existing_preparation(args, target_tokens, request):
    """Reject incomplete or incompatible outputs before allowing a resume."""
    manifest_path = args.out_dir / 'selection_manifest.json'
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if not isinstance(manifest, dict):
            raise ValueError('manifest must contain an object')
        expected = dict(
            schema_version=2, timeline_mode='uniform', timeline_version=TIMELINE_VERSION,
            fps=12, split=f'nuScenes {args.split}', num_clips=request['num_clips'],
            future_frames=args.future_frames, warmup_frames=args.warmup_frames,
            seed=args.seed, excluded_locations=request['excluded_locations'],
        )
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f'{key} differs from the requested preparation')
        for key in ('lo_ann_file', 'ann_12hz'):
            if Path(manifest[key]).resolve() != getattr(args, key).resolve():
                raise ValueError(f'{key} differs from the requested preparation')
        recorded_request = manifest.get('preparation_request')
        if recorded_request is not None and recorded_request != request:
            raise ValueError('preparation options or source inputs have changed')
        records = manifest['records']
        if (len(records) != request['num_clips'] or
                [row['clip_index'] for row in records] != list(range(len(records)))):
            raise ValueError('clip records are incomplete or out of order')
        if target_tokens and [row['target_token'] for row in records] != target_tokens:
            raise ValueError('target tokens differ from the requested preparation')
        annotation = args.out_dir / 'selected_12hz_frames.pkl'
        if sha256_file(annotation) != manifest['compact_annotation_sha256']:
            raise ValueError('selected annotations failed SHA256 verification')
        for row in records:
            if (row['frames'] != args.future_frames + 1 or
                    row['timeline_version'] != TIMELINE_VERSION or
                    row['generation_seed'] != args.seed + row['clip_index']):
                raise ValueError(f"clip {row['clip_index']} has incompatible frame/seed metadata")
            condition = (args.out_dir / row['condition_file']).resolve()
            if not condition.is_relative_to(args.out_dir.resolve()):
                raise ValueError('condition path is outside the output directory')
            if sha256_file(condition) != row['condition_sha256']:
                raise ValueError(f"condition {row['clip_index']} failed SHA256 verification")
        return manifest
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(
            f'Cannot reuse {manifest_path}: {exc}. '
            'Use a different --out-dir or pass --overwrite to rebuild conditions.'
        ) from exc


def main() -> None:
    args = parse_args()
    if args.num_clips <= 0:
        raise ValueError("--num-clips must be positive")
    conditions_dir = args.out_dir / "conditions"
    manifest_path = args.out_dir / "selection_manifest.json"
    compact_ann_path = args.out_dir / "selected_12hz_frames.pkl"
    target_tokens = requested_target_tokens(args)
    request = preparation_request(args, target_tokens)
    existing = None
    if manifest_path.exists() and not args.overwrite:
        existing = validate_existing_preparation(args, target_tokens, request)
        if existing.get('preparation_request') is not None:
            print(f'Reusing prepared conditions: {args.out_dir}', flush=True)
            return
        # Earlier raw preparations record selection but not every option.
        # Recompute that selection before trusting their completed artifacts.

    print(f"Loading LO annotation: {args.lo_ann_file}", flush=True)
    lo_infos, native_classes = load_infos(str(args.lo_ann_file))
    print(f"Loading 12 Hz annotation: {args.ann_12hz}", flush=True)
    infos12, classes12 = load_infos(str(args.ann_12hz))
    if target_tokens:
        candidates = explicit_candidates(
            lo_infos, infos12, target_tokens, args.future_frames)
    else:
        candidates = eligible_candidates(
            lo_infos, infos12, args.future_frames, args.warmup_frames)
    excluded_locations = set(args.exclude_locations)
    if excluded_locations:
        candidates = [
            row for row in candidates
            if row.get("location") not in excluded_locations
        ]
    if target_tokens:
        candidates_by_token = {
            row["target_token"]: row for row in candidates
        }
        missing = [
            token for token in target_tokens
            if token not in candidates_by_token
        ]
        if missing:
            raise ValueError(
                f"Explicit target tokens are not eligible: {missing}")
        selected = []
        for token in target_tokens:
            row = dict(candidates_by_token[token])
            row["motion_category"] = "explicit_target"
            selected.append(row)
    else:
        selected = select_diverse(
            candidates, args.num_clips, args.turn_threshold_deg,
            args.low_motion_threshold_m, args.seed,
            allow_scene_repeats=args.allow_scene_repeats)

    if existing is not None:
        if (len(existing['records']) != len(selected) or
                any(any(record.get(key) != value for key, value in selection.items())
                    for record, selection in zip(existing['records'], selected))):
            raise ValueError(
                'Existing clip selection differs from the requested preparation; '
                'use a different --out-dir or pass --overwrite to rebuild conditions.')
        print(f'Reusing prepared conditions: {args.out_dir}', flush=True)
        return

    scene_locations = load_scene_locations(str(args.data_root), args.version)
    if args.map_geometry_cache == 'spatial':
        from tools.trajectory_generation.map_geometry_cache import SpatialBEVLayoutRenderer
        renderer = SpatialBEVLayoutRenderer(str(args.data_root))
    else:
        renderer = BEVLayoutRenderer(str(args.data_root))
    conditions_dir.mkdir(parents=True, exist_ok=True)
    compact_clips = []
    records = []
    raw_index = None
    implementation = {str(p): sha256_file(p) for p in (
        Path(__file__), Path('tools/trajectory_generation/resample_scene_timeline.py'),
        Path('tools/trajectory_generation/bev_layout.py'),
        Path('tools/trajectory_generation/box_conventions.py'))}
    if args.map_geometry_cache == 'spatial':
        implementation['map_geometry_cache'] = sha256_file('tools/trajectory_generation/map_geometry_cache.py')
    for clip_index, selection in enumerate(tqdm(selected, desc="Render 12 Hz BEV")):
        start = selection["interp_index"]
        frames = infos12[start:start + args.future_frames + 1]
        if frames[1].get('timeline_version') == TIMELINE_VERSION:
            validate_timeline(frames)
        else:
            if raw_index is None:
                from tools.trajectory_generation.raw_timeline_index import RawTimelineIndex
                raw_index = RawTimelineIndex(args.data_root, args.timeline_cache_dir, args.version)
            native_start = lo_infos[int(selection['lo_dataset_index'])]
            frames, _, _ = resample_frames(frames[0], native_start, native_classes,
                                           raw_index.scene(str(frames[0]['token'])),
                                           args.data_root, args.future_frames)
        clock_version = TIMELINE_VERSION
        location = get_info_location(frames[0], scene_locations)
        timeline_signature = identity_key(dict(
            clock_version=clock_version,
            frames_sha256=hashlib.sha256(pickle.dumps(frames, protocol=4)).hexdigest(),
            classes=list(classes12), implementation=implementation,
            map_geometry_cache=args.map_geometry_cache,
            map=file_identity(args.data_root / 'maps/expansion' / f'{location}.json')))
        target_info = frames[0]
        future_infos = frames[1:]
        condition_relpath = Path("conditions") / f"{clip_index:06d}.npz"
        future_tokens = [str(info["token"]) for info in future_infos]
        condition_path = args.out_dir / condition_relpath
        if not valid_cached_condition(
                condition_path, str(selection["target_token"]),
                future_tokens, args.future_frames, timeline_signature):
            current_pose = get_ego2global(target_info)
            future_poses = np.stack([
                get_ego2global(info) for info in future_infos])
            target_lidar2ego = get_lidar2ego(target_info)
            render_infos = [keyframe_info_to_render_info(
                info, scene_locations, classes12, box_convention='legacy')
                for info in future_infos]
            location = get_info_location(target_info, scene_locations)
            layouts = render_layout_sequence(
                renderer, future_poses, target_lidar2ego, render_infos, location)
            curr2futu = current_to_future_transforms(
                current_pose, future_poses)
            with atomic_file(condition_path) as stream:
                np.savez_compressed(
                    stream,
                    schema_version=np.asarray(2, dtype=np.int64),
                    timeline_version=np.asarray(clock_version),
                    timeline_signature=np.asarray(timeline_signature),
                    frame_timestamps_us=np.asarray([f['timestamp'] for f in frames]),
                    target_token=np.asarray(selection["target_token"]),
                    scene_name=np.asarray(selection["interp_scene_name"]),
                    future_tokens=np.asarray(future_tokens),
                    candidate_curr2futu_ego=curr2futu[None].astype(np.float32),
                    bev_layout=layouts[None].astype(np.uint8),
                )
        record = dict(selection)
        record.update(
            clip_index=clip_index,
            condition_file=str(condition_relpath),
            generation_seed=args.seed + clip_index,
            frames=args.future_frames + 1,
            timeline_version=clock_version, timeline_signature=timeline_signature,
            condition_sha256=sha256_file(condition_path),
        )
        records.append(record)
        compact_clips.append(dict(record=record, frames=frames))

    with atomic_file(compact_ann_path) as file:
        pickle.dump(dict(clips=compact_clips), file, protocol=pickle.HIGHEST_PROTOCOL)
    if raw_index is not None:
        raw_index.close()
    counts = defaultdict(int)
    for record in records:
        counts[record["motion_category"]] += 1
    manifest = dict(
        schema_version=2,
        preparation_request=request,
        timeline_mode='uniform',
        timeline_version=TIMELINE_VERSION,
        compact_annotation_sha256=sha256_file(compact_ann_path),
        split=f"nuScenes {args.split}",
        num_clips=len(records),
        fps=12,
        future_frames=args.future_frames,
        warmup_frames=args.warmup_frames,
        seed=args.seed,
        lo_ann_file=str(args.lo_ann_file.resolve()),
        ann_12hz=str(args.ann_12hz.resolve()),
        excluded_locations=sorted(excluded_locations),
        unique_scenes=len({row["lo_scene_name"] for row in records}),
        unique_target_tokens=len({row["target_token"] for row in records}),
        compact_annotation=compact_ann_path.name,
        motion_category_counts=dict(counts),
        missing_dynamic_frames=sum(
            len(row.get("missing_dynamic_offsets", [])) for row in records),
        records=records,
    )
    with atomic_file(manifest_path, 'w') as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
    print(json.dumps({
        "manifest": str(manifest_path),
        "clips": len(records),
        "motion_categories": dict(counts),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
