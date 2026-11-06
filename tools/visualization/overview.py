#!/usr/bin/env python3
"""Compose generated surround videos and matching OCC/LiDAR into a scene GIF."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import logging
from pathlib import Path
import sys
import tempfile
import time

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.genie.occ_generation.scene_runtime import atomic_file, sha256_file, utc_now, write_json
from tools.visualization.surround import CAMERA_GRID, camera_grid, label_font
from tools.visualization.gif import encode_gif

logger = logging.getLogger('scene_overview')


def matching_occ_preview(scene_dir, video):
    """Use the recorded generation run, independent of the current latest link."""
    run_id = video.get('source_run_id')
    if run_id:
        if Path(run_id).name != run_id:
            raise ValueError('Invalid source run ID in the video manifest')
        run_dir = scene_dir / 'runs' / run_id
        prediction = run_dir / 'predictions' / 'sequence.npz'
        preview = run_dir / 'visualization' / 'overview.gif'
        metadata_path = run_dir / 'visualization' / 'render_meta.json'
    else:
        # Legacy predictions are supported only when their render provenance is
        # available. Never pick a different scene version by filename recency.
        prediction = Path(video['predictions']) if video.get('predictions') else scene_dir / 'predictions.npz'
        preview = scene_dir / 'visualization_3d.gif'
        metadata_path = preview.with_suffix('.json')
    expected = video.get('prediction_sha256')
    if not expected or sha256_file(prediction) != expected:
        raise ValueError('Video and OCC/LiDAR prediction sources do not match')
    if not preview.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f'Missing OCC/LiDAR preview or metadata: {preview}. '
            'Create the preview with tools/visualization/scene.py '
            f'in the loworld environment, using prediction {prediction}, '
            f'output {preview}, and --metadata {metadata_path}.')
    metadata = json.loads(metadata_path.read_text())
    if metadata.get('source_sha256') != expected:
        raise ValueError(f'OCC/LiDAR render provenance does not match this video: {metadata_path}')
    if not preview.is_file():
        raise FileNotFoundError(preview)
    return preview


def render_overview(scene_dir, *, tile_width=640):
    started = time.perf_counter()
    scene_dir = Path(scene_dir).resolve()
    video_dir = scene_dir / 'video'
    video_manifest = video_dir / 'manifest.json'
    video = json.loads(video_manifest.read_text())
    if video.get('status') != 'completed':
        raise ValueError('Complete six-camera video generation before composing an overview')
    if tile_width < 1:
        raise ValueError('Tile width must be positive')
    count, fps = int(video['frames']), float(video['fps'])
    if count < 1 or not 1 <= fps <= 100:
        raise ValueError('Expected a positive frame count and GIF FPS between 1 and 100')
    try:
        occ_path = matching_occ_preview(scene_dir, video)
    except BaseException as error:
        video.update(overview_status='failed', overview_error=str(error))
        write_json(video_manifest, video)
        write_json(scene_dir / 'overview.json', dict(
            status='failed', source_run_id=video.get('source_run_id'), video_run_id=video.get('run_id'),
            error=str(error), error_type=type(error).__name__, finished_at=utc_now()))
        raise
    paths = {cam: video_dir / f'000000_{cam}.mp4' for row in CAMERA_GRID for cam in row}
    sources = [occ_path, *paths.values()]
    identities = {str(path): sha256_file(path) for path in sources}
    metadata_path = scene_dir / 'overview.json'
    metadata = dict(status='running', started_at=utc_now(), video_run_id=video.get('run_id'),
                    source_run_id=video.get('source_run_id'),
                    prediction_sha256=video['prediction_sha256'],
                    source_sha256=identities, frame_alignment='same frame index; frame 0 is current reconstruction',
                    camera_grid=CAMERA_GRID, mirrored=False,
                    layout='generated six-camera surround above; OCC and LiDAR below',
                    overview='overview.gif', surround='video/surround.gif')
    video['overview_status'] = 'running'
    write_json(video_manifest, video)
    write_json(metadata_path, metadata)
    try:
        logger.info('Compose scene overview: %d frames at %.3g FPS', count, fps)
        logger.info('Matched OCC/LiDAR source: %s', occ_path)
        with ExitStack() as stack:
            occ = stack.enter_context(Image.open(occ_path))
            if occ.n_frames < count:
                raise ValueError(f'OCC/LiDAR has {occ.n_frames} frames, videos require {count}')
            captures = {}
            video_sizes = set()
            for cam, path in paths.items():
                capture = cv2.VideoCapture(str(path))
                stack.callback(capture.release)
                if not capture.isOpened():
                    raise RuntimeError(f'Cannot open {path}')
                found_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                found_fps = capture.get(cv2.CAP_PROP_FPS)
                if found_count != count or not np.isclose(found_fps, fps, atol=.01):
                    raise ValueError(f'{path}: expected {count} frames at {fps} FPS, '
                                     f'got {found_count} at {found_fps}')
                video_sizes.add((int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                 int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
                captures[cam] = capture
            if len(video_sizes) != 1:
                raise ValueError('Camera video resolutions do not match')
            width, height = video_sizes.pop()
            if video.get('resolution') and [height, width] != video['resolution']:
                raise ValueError('Camera video dimensions do not match the video manifest')
            metadata['source_video_size'] = [width, height]
            tile_height = max(1, round(tile_width * height / width))
            surround_size = (tile_width * 3 + 48, tile_height * 2 + 36)
            occ_height = max(1, round(surround_size[0] * occ.height / occ.width))
            overview_size = (surround_size[0], surround_size[1] + 12 + occ_height)
            scratch = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='.overview_', dir=video_dir)))
            surround_frames = np.memmap(scratch / 'surround.raw', dtype=np.uint8, mode='w+',
                                       shape=(count, surround_size[1], surround_size[0], 3))
            overview_frames = np.memmap(scratch / 'overview.raw', dtype=np.uint8, mode='w+',
                                       shape=(count, overview_size[1], overview_size[0], 3))
            stack.callback(surround_frames._mmap.close)
            stack.callback(overview_frames._mmap.close)
            font = label_font()
            for index in range(count):
                images = {}
                for cam, capture in captures.items():
                    ok, frame = capture.read()
                    if not ok:
                        raise RuntimeError(f'Cannot decode frame {index} from {paths[cam]}')
                    images[cam] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                surround = camera_grid(images, tile_width, tile_height, font)
                occ.seek(index)
                occ_frame = occ.convert('RGB').resize((overview_size[0], occ_height), Image.Resampling.LANCZOS)
                overview = Image.new('RGB', overview_size, 'white')
                overview.paste(surround, (0, 0))
                overview.paste(occ_frame, (0, surround_size[1] + 12))
                surround_frames[index] = np.asarray(surround)
                overview_frames[index] = np.asarray(overview)
                if index == 0 or (index + 1) % 10 == 0 or index + 1 == count:
                    logger.info('Composed frame %d/%d', index + 1, count)
            logger.info('Encode generated surround GIF')
            surround_stats = encode_gif(surround_frames, video_dir / 'surround.gif', fps)
            logger.info('Encode scene overview GIF')
            overview_stats = encode_gif(overview_frames, scene_dir / 'overview.gif', fps)
        if any(sha256_file(path) != identities[str(path)] for path in sources):
            raise RuntimeError('A video or OCC/LiDAR source changed while composing the overview')
    except BaseException as error:
        metadata.update(status='failed', error=str(error), error_type=type(error).__name__)
        video.update(overview_status='failed', overview_error=str(error))
        raise
    else:
        metadata.update(status='completed', animation=overview_stats, surround_animation=surround_stats)
        video.update(overview_status='completed', overview='../overview.gif',
                     overview_metadata='../overview.json', surround='surround.gif')
        video.pop('overview_error', None)
        write_json(video_dir / 'surround_meta.json', dict(
            source_run_id=video.get('source_run_id'), video_run_id=video.get('run_id'),
            camera_grid=CAMERA_GRID, mirrored=False, animation=surround_stats,
            source_sha256={str(path): identities[str(path)] for path in paths.values()}))
        add_preview_links(scene_dir / 'README.md', overview='overview.gif', surround='video/surround.gif')
        add_preview_links(video_dir / 'README.md', overview='../overview.gif', surround='surround.gif')
    finally:
        metadata.update(finished_at=utc_now(), wall_seconds=time.perf_counter() - started)
        write_json(metadata_path, metadata)
        write_json(video_manifest, video)
    logger.info('Saved scene overview: %s (%.1f s)', scene_dir / 'overview.gif', metadata['wall_seconds'])
    return metadata


def add_preview_links(path, *, overview, surround):
    marker = '## Generated video overview'
    existing = path.read_text() if path.exists() else '# Scene results\n'
    start, end = '<!-- generated-overview:start -->', '<!-- generated-overview:end -->'
    section = (marker + '\n\n'
               f'- [Overview: surround + OCC/LiDAR]({overview})\n'
               f'- [Generated surround video GIF]({surround})\n\n'
               'Front row: FRONT LEFT / FRONT / FRONT RIGHT. '
               'Back row: BACK LEFT / BACK / BACK RIGHT. No camera mirroring.\n')
    block = start + '\n' + section + end
    if start in existing and end in existing:
        before, tail = existing.split(start, 1)
        _, after = tail.split(end, 1)
        updated = before + block + after
    elif section in existing:
        updated = existing.replace(section, block + '\n', 1)
    else:
        updated = existing.rstrip() + '\n\n' + block + '\n'
    with atomic_file(path, 'w') as stream:
        stream.write(updated)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-dir', type=Path, required=True)
    parser.add_argument('--tile-width', type=int, default=640)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
                        datefmt='%H:%M:%S')
    render_overview(args.scene_dir, tile_width=args.tile_width)


if __name__ == '__main__':
    main()
