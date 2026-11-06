#!/usr/bin/env python3
"""Render six-camera semantic conditions from a scene directory or prediction NPZ.

The output follows the GenieDrive dataset contract: one RGB video
and one semantic-control video for each of the six nuScenes cameras, plus a
single-record meta.json.  Frame 0 is the current latent reconstruction and
subsequent frames are autoregressive LO-World predictions.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.genie.occ_generation.scene_runtime import atomic_file, sha256_file, write_json
from tools.genie.semantic_export.scene_inputs import scene_defaults, prediction_annotation, validate_prediction_clock
from tools.genie.semantic_export.palette import CAM_NAMES, GENIE_COLOR_ID


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--scene-dir', type=Path,
                        help='Generated scene directory; auto-select its latest successful prediction')
    source.add_argument('--predictions', type=Path,
                        help='Saved scene NPZ with its accompanying frozen timeline')
    parser.add_argument('--ann-file', type=Path,
                        help='Optional copy of the prediction timeline; must have identical content')
    parser.add_argument('--data-root', type=Path, default=REPO_ROOT / 'data/nuscenes')
    parser.add_argument('--out-dir', type=Path,
                        help='Defaults to <scene-dir>/conditions in scene mode')
    parser.add_argument("--fps", type=int, choices=(12,), default=12)
    parser.add_argument("--future-frames", type=int,
                        help='Export this many future frames; defaults to the complete prediction')
    parser.add_argument("--height", type=int, default=450)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--gaussian-scale", type=float, default=0.2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.scene_dir:
        args.scene_dir = args.scene_dir.resolve()
        selected = scene_defaults(args.scene_dir)
        args.predictions = selected['predictions']
        args.source_run_id = selected['source_run_id']
        args.prediction_sha256 = selected['prediction_sha256']
        if args.out_dir is None:
            args.out_dir = selected['out_dir']
    else:
        if args.out_dir is None:
            parser.error('--out-dir is required with --predictions')
        args.source_run_id = None
        args.prediction_sha256 = sha256_file(args.predictions)
    if args.future_frames is not None and args.future_frames < 1:
        parser.error('--future-frames must be positive')
    if args.height < 1 or args.width < 1 or args.gaussian_scale <= 0:
        parser.error('Image dimensions and Gaussian scale must be positive')
    for key in ('predictions', 'data_root', 'out_dir'):
        setattr(args, key, getattr(args, key).resolve())
    if args.out_dir == args.scene_dir or args.out_dir == args.predictions.parent.parent:
        parser.error('--out-dir must be a separate conditions directory, not a scene/run root')
    frozen = prediction_annotation(args.predictions)
    with np.load(args.predictions, allow_pickle=False) as archive:
        args.annotation_sha256 = str(archive['timeline_annotation_sha256'])
    if args.ann_file is not None and sha256_file(args.ann_file) != args.annotation_sha256:
        parser.error('--ann-file differs from the timeline recorded in this prediction')
    args.ann_file = frozen
    return args


def load_infos(path: Path) -> list[dict]:
    with path.open("rb") as file:
        data = pickle.load(file)
    infos = data['data_list']
    if not isinstance(infos, list):
        raise TypeError(f"Unsupported annotation structure in {path}")
    return infos


def load_prediction_clip(path: Path, future_frames: int | None):
    with np.load(path, allow_pickle=False) as data:
        target_token = str(data['target_token'].item())
        current = data['current_occ']
        future = data['pred_occ_12hz']
        future_tokens = data['future_tokens_12hz'].astype(str)
    if current.shape != (200, 200, 16):
        raise ValueError(f"Unexpected current OCC shape: {current.shape}")
    if future.shape[1:] != current.shape:
        raise ValueError(f"Unexpected future OCC shape: {future.shape}")
    if current.dtype != np.uint8 or future.dtype != np.uint8:
        raise ValueError('Prediction occupancy must use uint8 class labels')
    if future_tokens.shape != (len(future),):
        raise ValueError('Prediction future tokens do not match its OCC frame count')
    if future_frames is None:
        future_frames = len(future)
    if future_frames < 1:
        raise ValueError('Prediction contains no future OCC frames')
    if len(future) < future_frames:
        raise ValueError(
            f"Need {future_frames} future OCC frames, only found {len(future)}"
        )
    occ = np.concatenate([current[None], future[:future_frames]], axis=0)
    return target_token, occ, future_tokens[:future_frames]


def select_annotation_frames(
    infos: list[dict], target_token: str, future_frames: int
) -> tuple[int, list[dict]]:
    index = next(
        (idx for idx, info in enumerate(infos) if str(info['token']) == target_token),
        None,
    )
    if index is None:
        raise ValueError(f"Target token {target_token} was not found in annotation")
    frames = infos[index : index + future_frames + 1]
    if len(frames) != future_frames + 1:
        raise ValueError("Not enough annotation frames after the target token")
    scene_name = frames[0]['scene_name']
    if any(frame['scene_name'] != scene_name for frame in frames):
        raise ValueError("Requested clip crosses a scene boundary")
    return index, frames


def resolve_image_path(raw_path: str, data_root: Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        if path.parts[:2] == ('data', 'nuscenes'):
            path = Path(*path.parts[2:])
        path = data_root / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def make_writer(path: Path, size: tuple[int, int], fps: int) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    return writer


def create_full_center_coords(device: torch.device) -> torch.Tensor:
    shape = (200, 200, 16)
    x = torch.linspace(-40.0, 40.0, shape[0], device=device)
    y = torch.linspace(-40.0, 40.0, shape[1], device=device)
    z = torch.linspace(-1.0, 5.4, shape[2], device=device)
    grid = torch.meshgrid(x, y, z, indexing="ij")
    return torch.stack(grid, dim=-1).reshape(-1, 3).float()


def official_occ_remap(occ: np.ndarray) -> np.ndarray:
    """Apply GenieDrive's exact OCC3D-to-render-feature remapping."""
    mapped = np.asarray(occ, dtype=np.uint8).copy()
    empty_mask = mapped == 17
    mapped[mapped == 0] = 17
    mapped[empty_mask] = 0
    return mapped


@torch.inference_mode()
def render_occ_frame(
    render_fn,
    occ: np.ndarray,
    frame_info: dict,
    xyz: torch.Tensor,
    rot: torch.Tensor,
    scale: torch.Tensor,
    device: torch.device,
    image_shape: tuple[int, int],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    mapped = official_occ_remap(occ)
    labels = torch.from_numpy(mapped).to(device=device).reshape(-1).long()
    occupied = labels != 0
    labels = labels[occupied]
    features = torch.nn.functional.one_hot(labels, num_classes=20).float()
    count = int(occupied.sum().item())
    rgb = torch.zeros((count, 3), device=device)
    opacity = torch.ones((count, 1), device=device)

    rendered_color = {}
    rendered_label = []
    for cam in CAM_NAMES:
        cam_info = frame_info["images"][cam]
        intrinsic = torch.as_tensor(
            cam_info["cam2img"], device=device, dtype=torch.float32
        ).clone()
        native_height, native_width = 900.0, 1600.0
        intrinsic[0] *= image_shape[1] / native_width
        intrinsic[1] *= image_shape[0] / native_height
        cam2ego = torch.as_tensor(
            cam_info["cam2ego"], device=device, dtype=torch.float32
        )
        package = render_fn(
            cam2ego,
            intrinsic,
            image_shape,
            xyz[occupied],
            rgb,
            features,
            rot[occupied],
            scale[occupied],
            opacity,
            bg_color=[0, 0, 0],
        )
        semantic = package["render_feat"].argmax(dim=0).to(torch.uint8)
        semantic_np = semantic.cpu().numpy()
        rendered_label.append(semantic_np)
        # GENIE_COLOR_ID is RGB; cv2.VideoWriter expects BGR.
        rendered_color[cam] = GENIE_COLOR_ID[semantic_np][..., ::-1].copy()
    return rendered_color, np.stack(rendered_label, axis=0)


def validate_future_tokens(expected: np.ndarray, frames: list[dict]) -> None:
    actual = np.asarray([str(frame["token"]) for frame in frames[1:]], dtype=str)
    if expected.shape != actual.shape or np.any(expected != actual):
        raise ValueError('Prediction and annotation future tokens are not aligned')


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Gaussian rendering requires a CUDA GPU")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    from third_party.geniedrive_occ_rasterizer.gaussian_renderer import render

    target_token, occ_frames, future_tokens = load_prediction_clip(
        args.predictions, args.future_frames
    )
    args.future_frames = len(occ_frames) - 1
    prediction_sha256 = args.prediction_sha256
    annotation_sha256 = args.annotation_sha256
    infos = load_infos(args.ann_file)
    annotation_index, frames = select_annotation_frames(
        infos, target_token, args.future_frames
    )
    validate_future_tokens(future_tokens, frames)
    validate_prediction_clock(args.predictions, frames, args.fps)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rgb_paths = {cam: args.out_dir / f"RGB_000000_{cam}.mp4" for cam in CAM_NAMES}
    control_paths = {
        cam: args.out_dir / f"SEMANTIC_000000_{cam}.mp4" for cam in CAM_NAMES
    }
    outputs = [*rgb_paths.values(), *control_paths.values()]
    if not args.overwrite and any(path.exists() for path in outputs):
        raise FileExistsError(
            f"Output already exists in {args.out_dir}; pass --overwrite to replace it"
        )

    rgb_writers = {}
    control_writers = {}
    semantic_labels = []
    xyz = create_full_center_coords(device)
    rot = torch.zeros((xyz.shape[0], 4), device=device)
    rot[:, 0] = 1.0
    scale = torch.full(
        (xyz.shape[0], 3), args.gaussian_scale, device=device, dtype=torch.float32
    )

    try:
        for cam in CAM_NAMES:
            control_writers[cam] = make_writer(control_paths[cam], (args.width, args.height), args.fps)
        for occ, frame_info in tqdm(zip(occ_frames, frames), total=len(frames), desc="GS render"):
            colors, labels = render_occ_frame(
                render,
                occ,
                frame_info,
                xyz,
                rot,
                scale,
                device,
                (args.height, args.width),
            )
            semantic_labels.append(labels)
            for cam in CAM_NAMES:
                image_info = frame_info["images"][cam]
                image_path = resolve_image_path(
                    image_info['img_path'],
                    args.data_root,
                )
                rgb = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if rgb is None:
                    raise RuntimeError(f"Failed to read {image_path}")
                if cam not in rgb_writers:
                    rgb_writers[cam] = make_writer(
                        rgb_paths[cam], (rgb.shape[1], rgb.shape[0]), args.fps
                    )
                rgb_writers[cam].write(rgb)
                control_writers[cam].write(colors[cam])
    finally:
        for writer in rgb_writers.values():
            writer.release()
        for writer in control_writers.values():
            writer.release()

    if sha256_file(args.predictions) != prediction_sha256:
        raise RuntimeError('Prediction source changed while rendering')
    if sha256_file(args.ann_file) != annotation_sha256:
        raise RuntimeError('Timeline annotation changed while rendering')
    semantic_path = args.out_dir / "rendered_semantic_labels.npz"
    with atomic_file(semantic_path) as file:
        np.savez_compressed(
            file, semantic=np.stack(semantic_labels, axis=0),
            camera_names=np.asarray(CAM_NAMES),
            tokens=np.asarray([str(frame["token"]) for frame in frames]))

    record = {"text": "", "type": "video"}
    for cam in CAM_NAMES:
        record[f"file_path_{cam}"] = rgb_paths[cam].name
        record[f"control_file_path_{cam}"] = control_paths[cam].name
    record.update(
        {
            "_customavwm_scene_name": frames[0]['scene_name'],
            "_customavwm_token": target_token,
            "_condition_source": "LO-World predicted OCC via official GenieDrive GS",
        }
    )
    write_json(args.out_dir / 'meta.json', [record])

    manifest = {
        'status': 'completed',
        'annotation_sha256': annotation_sha256,
        'frame_timestamps_us': [frame['timestamp'] for frame in frames],
        'source_run_id': args.source_run_id,
        'scene_dir': str(args.scene_dir) if args.scene_dir else None,
        'prediction_sha256': prediction_sha256,
        "predictions": str(args.predictions.resolve()),
        "ann_file": str(args.ann_file.resolve()),
        'data_root': str(args.data_root.resolve()),
        "target_token": target_token,
        "annotation_index": annotation_index,
        "scene_name": frames[0]['scene_name'],
        "frames": len(frames),
        "future_frames": args.future_frames,
        "fps": args.fps,
        "camera_names": list(CAM_NAMES),
        "condition_resolution": [args.height, args.width],
        "occ_shape": list(occ_frames.shape[1:]),
        "occ_coordinate": "each frame in its own timestamp ego frame",
        "camera_transform": "same-timestamp cam2ego; no extra ego compensation",
        "label_remap": "exact GenieDrive official OCC remap: free 17->0, others 0->17",
        "gaussian_scale": args.gaussian_scale,
        "semantic_labels": semantic_path.name,
        "meta_json": "meta.json",
    }
    write_json(args.out_dir / 'manifest.json', manifest)
    readme = [
        '# 六相机语义条件', '',
        f'场景：`{manifest["scene_name"]}`；来源运行：`{args.source_run_id or "显式指定的预测"}`。', '',
        f'共 {len(frames)} 帧，{args.fps} FPS，语义分辨率 {args.width}×{args.height}。', '',
        '- `SEMANTIC_000000_CAM_*.mp4`：预测 OCC 渲染的六路语义条件。',
        '- `RGB_000000_CAM_*.mp4`：对应六路原 始相机参考视频。',
        '- `rendered_semantic_labels.npz`：T,C,H,W 语义类别、相机名称及帧 token。',
        '- `meta.json`：下一步 GenieDrive 视频生成使用的输入索引。',
        '- `manifest.json`：预测 SHA256、源运行、标注和渲染参数。', '',
        '第 0 帧是当前重建，其余为未来预测；相机顺序：', '',
        ', '.join(CAM_NAMES), '',
        '语义类别是 GenieDrive 官方 remap 后的类别，与原 OCC3D 类别编号不同。', '',
        f'源预测：`{args.predictions}`', '',
        f'预测 SHA256：`{prediction_sha256}`', '',
    ]
    with atomic_file(args.out_dir / 'README.md', 'w') as file:
        file.write('\n'.join(readme))
    from tools.visualization.surround import render_surround
    render_surround(args.out_dir)
    print(f'Saved {len(frames)} frames for six cameras: {args.out_dir}')


if __name__ == "__main__":
    main()
