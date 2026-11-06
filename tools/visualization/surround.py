#!/usr/bin/env python3
"""Compose a 3x2 condition GIF from saved semantic labels, without CUDA."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.genie.occ_generation.scene_runtime import atomic_file, sha256_file, write_json
from tools.genie.semantic_export.palette import CAM_NAMES, GENIE_COLOR_ID
from tools.visualization.gif import encode_gif

CAMERA_GRID = (
    ('CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT'),
    ('CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'),
)


def label_font():
    for path in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                 '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf'):
        if Path(path).is_file():
            return ImageFont.truetype(path, 17)
    return ImageFont.load_default()


def camera_grid(images, tile_width, tile_height, font, *, resample=Image.Resampling.LANCZOS):
    """Compose the same six-camera arrangement for semantic and RGB previews."""
    gap = 12
    canvas = Image.new('RGB', (3 * tile_width + 4 * gap, 2 * tile_height + 3 * gap), 'white')
    draw = ImageDraw.Draw(canvas)
    for row, cameras in enumerate(CAMERA_GRID):
        for column, camera in enumerate(cameras):
            x, y = gap + column * (tile_width + gap), gap + row * (tile_height + gap)
            canvas.paste(images[camera].resize((tile_width, tile_height), resample), (x, y))
            title = camera.removeprefix('CAM_').replace('_', ' ')
            text_width = draw.textlength(title, font=font)
            draw.rounded_rectangle((x + 8, y + 8, x + text_width + 24, y + 34), radius=5, fill='white')
            draw.text((x + 16, y + 10), title, font=font, fill=(64, 64, 64))
    return canvas


def semantic_palette():
    # Include the exact class colors and grays for the white layout/camera tags.
    colors = GENIE_COLOR_ID.tolist() + [[v, v, v] for v in range(0, 256, 4)] + [[255, 255, 255]]
    colors.extend([[255, 255, 255]] * (256 - len(colors)))
    palette = Image.new('P', (1, 1))
    palette.putpalette([channel for color in colors for channel in color])
    return palette


def render_surround(conditions_dir, *, tile_width=640):
    """Preview the lossless render labels; never modify the model's control videos."""
    started = time.perf_counter()
    conditions_dir = Path(conditions_dir).resolve()
    source = conditions_dir / 'rendered_semantic_labels.npz'
    manifest_path = conditions_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if manifest.get('status', 'completed') != 'completed':
        raise ValueError('Conditions are not complete')
    if tile_width < 1:
        raise ValueError('Tile width must be positive')
    fps = manifest.get('fps', 12)
    if not 1 <= fps <= 100:
        raise ValueError('GIF FPS must be between 1 and 100')
    source_digest = sha256_file(source)
    with np.load(source, allow_pickle=False) as archive:
        labels = archive['semantic']
        cameras = list(map(str, archive['camera_names']))
        tokens = archive['tokens']
    if (labels.ndim != 4 or labels.shape[1] != len(cameras)
            or labels.shape[0] != len(tokens) or not len(tokens)
            or len(cameras) != 6 or set(cameras) != set(CAM_NAMES)):
        raise ValueError('Expected semantic labels [T,6,H,W] and one token per frame')
    if labels.dtype.kind not in 'iu' or labels.min() < 0 or labels.max() >= len(GENIE_COLOR_ID):
        raise ValueError('Semantic labels must use the official GenieDrive render classes')
    if manifest.get('frames', len(labels)) != len(labels):
        raise ValueError('Semantic labels and condition manifest disagree on frame count')
    height, width = labels.shape[2:]
    tile_height = max(1, round(tile_width * height / width))
    size = (3 * tile_width + 48, 2 * tile_height + 36)
    font = label_font()
    camera_indices = {camera: i for i, camera in enumerate(cameras)}
    output = conditions_dir / 'surround.gif'
    with ExitStack() as stack:
        scratch = Path(stack.enter_context(tempfile.TemporaryDirectory(
            prefix='.surround_', dir=conditions_dir)))
        storage = np.memmap(scratch / 'rgb.raw', dtype=np.uint8, mode='w+',
                            shape=(len(labels), size[1], size[0], 3))
        stack.callback(storage._mmap.close)
        for index, frame in enumerate(tqdm(labels, desc='Compose surround conditions', unit='frame')):
            images = {camera: Image.fromarray(GENIE_COLOR_ID[frame[index]])
                      for camera, index in camera_indices.items()}
            canvas = camera_grid(images, tile_width, tile_height, font,
                                 resample=Image.Resampling.NEAREST)
            storage[index] = np.asarray(canvas)
        animation = encode_gif(storage, output, fps, palette=semantic_palette())
    if sha256_file(source) != source_digest:
        raise RuntimeError('Semantic label source changed while composing the GIF')
    metadata = dict(source=str(source), source_sha256=source_digest,
                    source_run_id=manifest.get('source_run_id'),
                    prediction_sha256=manifest.get('prediction_sha256'),
                    camera_grid=CAMERA_GRID, mirrored=False,
                    color_source='lossless semantic labels with the official GenieDrive RGB palette',
                    resize='nearest', tile_size=[tile_width, tile_height],
                    animation=animation, wall_seconds=time.perf_counter() - started)
    write_json(conditions_dir / 'surround_meta.json', metadata)
    manifest['surround_preview'] = dict(gif='surround.gif', metadata='surround_meta.json',
                                        **animation)
    write_json(manifest_path, manifest)
    readme_path = conditions_dir / 'README.md'
    if readme_path.exists():
        readme = readme_path.read_text()
        if '[环视 condition GIF](surround.gif)' not in readme:
            with atomic_file(readme_path, 'w') as file:
                file.write(readme.rstrip() + '\n\n## 环视预览\n\n'
                           '[环视 condition GIF](surround.gif) · [预览参数](surround_meta.json)\n\n'
                           '上排：前左、前、前右；下排：后左、后、后右。各相机画面不做镜像。\n')
    print(f'Saved {output} ({animation["frame_count"]} frames, {fps} FPS)', flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--scene-dir', type=Path, help='Scene directory containing conditions/')
    source.add_argument('--conditions-dir', type=Path, help='Explicit existing condition directory')
    parser.add_argument('--tile-width', type=int, default=640)
    args = parser.parse_args()
    conditions_dir = args.conditions_dir or args.scene_dir / 'conditions'
    render_surround(conditions_dir, tile_width=args.tile_width)


if __name__ == '__main__':
    main()
