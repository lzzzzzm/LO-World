#!/usr/bin/env python3
"""Prepare generated nuScenes videos for WorldBench BEVFormer evaluation.

The official WorldBench downstream protocol reads frames 0, 6 and 12 from
each 12 Hz video, resizes them to the native nuScenes camera resolution, and
evaluates the resulting 3-keyframe sequences with BEVFormer.  Unlike the
stock WorldBench converter, this utility accepts clips that start at arbitrary
native nuScenes keyframes.
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fvd_progress import EvaluationProgress

import cv2
import numpy as np
from pyquaternion import Quaternion


CAM_NAMES = (
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
)
ROOT = Path(__file__).resolve().parents[4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--source-infos", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--nuscenes-sample-json", type=Path,
        default=ROOT / "data/nuscenes/v1.0-trainval/sample.json",
        help="Official nuScenes sample table used to recover native scene tokens.")
    parser.add_argument(
        "--frame-indices", type=int, nargs="+", default=(0, 6, 12),
        help="Generated-video frames used by the downstream protocol.")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--overwrite-images", action="store_true",
        help="Rewrite decoded PNGs as well as the annotation pickle.")
    return parser.parse_args()


class NumpyCompatibleUnpickler(pickle.Unpickler):
    """Load NumPy 2.x pickles in the NumPy 1.x WorldBench environment."""

    def find_class(self, module: str, name: str):
        if module == "numpy._core" or module.startswith("numpy._core."):
            module = "numpy.core" + module[len("numpy._core"):]
        return super().find_class(module, name)


def load_pickle(path: Path):
    with path.open("rb") as file:
        return NumpyCompatibleUnpickler(file).load()


def to_wcoda_info(source_info: dict) -> dict:
    """Convert the repository's MMEngine nuScenes info to W-CODA format."""
    if "cams" in source_info:
        return copy.deepcopy(source_info)
    if "images" not in source_info:
        raise KeyError("Source info has neither 'cams' nor 'images'")

    info = copy.deepcopy(source_info)
    lidar_points = info.get("lidar_points", {})
    lidar2ego = np.asarray(lidar_points["lidar2ego"], dtype=np.float64)
    ego2global = np.asarray(info["ego2global"], dtype=np.float64)
    info["lidar_path"] = lidar_points.get("lidar_path", "")
    info["sweeps"] = lidar_points.get("sweeps", [])
    info["lidar2ego_translation"] = lidar2ego[:3, 3].tolist()
    info["lidar2ego_rotation"] = Quaternion(
        matrix=lidar2ego[:3, :3]).elements.tolist()
    info["ego2global_translation"] = ego2global[:3, 3].tolist()
    info["ego2global_rotation"] = Quaternion(
        matrix=ego2global[:3, :3]).elements.tolist()

    cams = {}
    for cam_name, image in info["images"].items():
        lidar2cam = np.asarray(image["lidar2cam"], dtype=np.float64)
        cam2lidar = np.linalg.inv(lidar2cam)
        cams[cam_name] = {
            "data_path": image["img_path"],
            "sample_data_token": image.get("sample_data_token", ""),
            "timestamp": image.get("timestamp", info["timestamp"]),
            "sensor2lidar_rotation": cam2lidar[:3, :3],
            "sensor2lidar_translation": cam2lidar[:3, 3],
            "cam_intrinsic": np.asarray(image["cam2img"], dtype=np.float64),
        }
    info["cams"] = cams
    return info


def read_selected_frames(video_path: Path, indices: list[int]) -> dict[int, object]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    wanted = set(indices)
    frames = {}
    frame_index = 0
    while wanted:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index in wanted:
            frames[frame_index] = frame
            wanted.remove(frame_index)
        frame_index += 1
    capture.release()
    if wanted:
        raise ValueError(
            f"{video_path} is missing requested frames: {sorted(wanted)}")
    return frames


def clip_tokens(record: dict, dataset_dir: Path, frame_indices: list[int]) -> list[str]:
    condition_path = dataset_dir / record["condition_file"]
    with np.load(condition_path, allow_pickle=False) as condition:
        future = [str(token) for token in condition["future_tokens"]]
    all_tokens = [str(record["target_token"]), *future]
    if max(frame_indices) >= len(all_tokens):
        raise IndexError(
            f"Clip {record['clip_index']} has {len(all_tokens)} frames, "
            f"requested {max(frame_indices)}")
    return [all_tokens[index] for index in frame_indices]


def main() -> None:
    args = parse_args()
    progress = EvaluationProgress("WorldBench prepare")
    frame_indices = list(args.frame_indices)
    if sorted(set(frame_indices)) != frame_indices or frame_indices[0] < 0:
        raise ValueError("--frame-indices must be sorted, unique, and non-negative")

    with progress.stage("load clip selection and source annotations"):
        manifest = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
        source = load_pickle(args.source_infos)
    if isinstance(source, dict):
        if "infos" in source:
            source_infos = source["infos"]
        elif "data_list" in source:
            source_infos = source["data_list"]
        else:
            raise KeyError(
                f"{args.source_infos} has neither 'infos' nor 'data_list'")
    else:
        source_infos = source
    source_by_token = {str(info["token"]): info for info in source_infos}
    progress.info("Loaded %d clips and %d source frames; selected video frames: %s",
                  len(manifest["records"]), len(source_infos), frame_indices)
    sample_table = json.loads(
        args.nuscenes_sample_json.read_text(encoding="utf-8"))
    sample_to_scene = {
        str(sample["token"]): str(sample["scene_token"])
        for sample in sample_table
    }

    images_dir = args.out_dir / "images"
    annotation_path = args.out_dir / "nuscenes_infos_val_generated_3keyframes.pkl"
    summary_path = args.out_dir / "prepare_summary.json"
    if annotation_path.exists() and not args.overwrite:
        raise FileExistsError(f"{annotation_path} exists; pass --overwrite")
    images_dir.mkdir(parents=True, exist_ok=True)

    eval_infos = []
    started = time.monotonic()
    written_images = reused_images = 0
    with progress.stage("prepare generated frames and annotations"):
        for record_index, record in enumerate(manifest["records"], 1):
            clip_index = int(record["clip_index"])
            tokens = clip_tokens(record, args.dataset_dir, frame_indices)
            target_token = str(record["target_token"])
            if target_token not in sample_to_scene:
                raise KeyError(
                    f"Target token {target_token} is absent from "
                    f"{args.nuscenes_sample_json}")
            scene_token = f"{sample_to_scene[target_token]}_gen0"
            infos = []
            for local_index, token in enumerate(tokens):
                if token not in source_by_token:
                    raise KeyError(f"Token {token} is absent from {args.source_infos}")
                info = to_wcoda_info(source_by_token[token])
                info["scene_token"] = scene_token
                # Some interpolated 12 Hz infos omit location.  Keep this
                # explicit so the W-CODA dataset never needs to resolve our
                # generated scene suffix through the native nuScenes tables.
                info["location"] = str(record["location"])
                # BEVFormer fills pose and yaw into this 18-D nuScenes CAN bus
                # vector in get_data_info(), but the key must already exist.
                # Keep NumPy's default float64: the legacy torchvision rotate()
                # accepts np.float64 yaw values but rejects np.float32 scalars.
                info["can_bus"] = np.zeros(18)
                for cam_info in info["cams"].values():
                    if "cam_intrinsic" not in cam_info:
                        cam_info["cam_intrinsic"] = np.asarray(
                            cam_info["camera_intrinsics"])
                info["frame_idx"] = local_index
                info["prev"] = tokens[local_index - 1] if local_index else ""
                info["next"] = (
                    tokens[local_index + 1]
                    if local_index + 1 < len(tokens) else "")
                infos.append(info)

            for cam in CAM_NAMES:
                video_path = args.samples_dir / f"{clip_index:06d}_{cam}.mp4"
                cam_dir = images_dir / cam
                cam_dir.mkdir(parents=True, exist_ok=True)
                image_paths = [
                    cam_dir / f"{clip_index:06d}_f{index:02d}_{cam}.png"
                    for index in frame_indices
                ]
                decoded = None
                if args.overwrite_images or any(
                        not path.exists() for path in image_paths):
                    decoded = read_selected_frames(video_path, frame_indices)
                for local_index, generated_index in enumerate(frame_indices):
                    image_path = image_paths[local_index]
                    if args.overwrite_images or not image_path.exists():
                        frame = cv2.resize(
                            decoded[generated_index], (args.width, args.height),
                            interpolation=cv2.INTER_LINEAR)
                        if not cv2.imwrite(str(image_path), frame):
                            raise RuntimeError(f"Failed to write {image_path}")
                        written_images += 1
                    else:
                        reused_images += 1
                    infos[local_index]["cams"][cam]["data_path"] = str(
                        image_path.resolve())

            eval_infos.extend(infos)
            if record_index == 1 or record_index % 10 == 0 or record_index == len(manifest["records"]):
                elapsed = time.monotonic() - started
                progress.info(
                    "Prepared clips %d/%d; frames %d; images written=%d, reused=%d; elapsed %.1fs; ETA %.1fs",
                    record_index, len(manifest["records"]), len(eval_infos), written_images,
                    reused_images, elapsed,
                    elapsed / record_index * (len(manifest["records"]) - record_index))

    # WorldBench's BEVFormer evaluator expects the native nuScenes version
    # even when poses between native keyframes were interpolated.
    output = {
        "infos": eval_infos,
        "metadata": {"version": "v1.0-trainval"},
    }
    with annotation_path.open("wb") as file:
        pickle.dump(output, file, protocol=pickle.HIGHEST_PROTOCOL)

    summary = {
        "samples_dir": str(args.samples_dir.resolve()),
        "annotation": str(annotation_path.resolve()),
        "num_clips": len(manifest["records"]),
        "frame_indices": frame_indices,
        "num_eval_frames": len(eval_infos),
        "num_images": len(eval_infos) * len(CAM_NAMES),
        "image_size": [args.width, args.height],
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    progress.info("Preparation completed: %d frames, %d images; annotation: %s",
                  len(eval_infos), summary["num_images"], annotation_path)
    progress.info("Preparation summary: %s", summary_path)


if __name__ == "__main__":
    main()
