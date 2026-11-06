#!/usr/bin/env python3
"""Generate a scene's six-camera videos using the current Python environment."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.genie.occ_generation.scene_outputs import run_logged_process
from tools.genie.occ_generation.scene_runtime import (
    atomic_file, file_identity, sha256_file, utc_now, write_json,
)

CAMERAS = ('CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
           'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT')
logger = logging.getLogger('generate_video')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-dir', type=Path, required=True)
    parser.add_argument('--gpu', default='0', help='One physical GPU ID (default: 0)')
    parser.add_argument('--base-model', type=Path,
                        default=ROOT / 'ckpts/geniedrive/Wan2.1-Fun-V1.1-1.3B-Control')
    parser.add_argument('--checkpoint', type=Path,
                        default=ROOT / 'ckpts/geniedrive/geniedrive_448x800.safetensors')
    parser.add_argument('--height', type=int, default=448)
    parser.add_argument('--width', type=int, default=800)
    parser.add_argument('--num-inference-steps', type=int, default=50)
    parser.add_argument('--guidance-scale', type=float, default=6.0)
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--gpu-memory-mode', default='model_cpu_offload', choices=(
        'model_full_load', 'model_cpu_offload', 'sequential_cpu_offload',
        'model_full_load_and_qfloat8', 'model_cpu_offload_and_qfloat8'))
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)
    if not args.gpu.isdigit():
        parser.error('--gpu must specify one nonnegative GPU index')
    if min(args.height, args.width, args.num_inference_steps) < 1:
        parser.error('Image dimensions and inference steps must be positive')
    for key in ('scene_dir', 'base_model', 'checkpoint'):
        setattr(args, key, getattr(args, key).resolve())
    return args


def run_scene(args):
    logger.info('Read scene conditions: %s', args.scene_dir / 'conditions')
    conditions_dir = args.scene_dir / 'conditions'
    condition_manifest = conditions_dir / 'manifest.json'
    meta_path = conditions_dir / 'meta.json'
    conditions = json.loads(condition_manifest.read_text())
    if conditions.get('status') != 'completed':
        raise ValueError('Generate complete conditions before starting video generation')
    frame_count, fps = int(conditions['frames']), int(conditions['fps'])
    if frame_count < 1 or (frame_count - 1) % 4 or fps < 1:
        raise ValueError('Wan requires a 4n+1 frame count and positive FPS; '
                         f'conditions contain {frame_count} frames at {fps} FPS')
    records = json.loads(meta_path.read_text())
    if len(records) != 1:
        raise ValueError('Scene video generation requires exactly one condition record')
    inputs = []
    for camera in CAMERAS:
        for key in (f'file_path_{camera}', f'control_file_path_{camera}'):
            inputs.append(file_identity(conditions_dir / records[0][key]))
    if not args.base_model.is_dir():
        raise FileNotFoundError(f'Wan base model missing: {args.base_model}; use --base-model')
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    logger.info('Conditions ready: %d frames, %d FPS, six cameras', frame_count, fps)
    logger.info('Video inference Python: %s', sys.executable)
    output = args.scene_dir / 'video'
    expected = [output / f'000000_{camera}.mp4' for camera in CAMERAS]
    if not args.overwrite and any(path.exists() for path in expected):
        raise FileExistsError(f'Videos already exist in {output}; use --overwrite to regenerate')
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(status='running', run_id=uuid.uuid4().hex, started_at=utc_now(),
                    scene_dir=str(args.scene_dir), source_run_id=conditions.get('source_run_id'),
                    prediction_sha256=conditions.get('prediction_sha256'),
                    conditions=str(conditions_dir), condition_manifest_sha256=sha256_file(condition_manifest),
                    condition_meta_sha256=sha256_file(meta_path), condition_inputs=inputs,
                    frames=frame_count, fps=fps, resolution=[args.height, args.width],
                    base_model=str(args.base_model), checkpoint=str(args.checkpoint),
                    python_executable=sys.executable, conda_prefix=os.environ.get('CONDA_PREFIX'),
                    parameters=vars(args))
    overrides = dict(CUDA_VISIBLE_DEVICES=args.gpu, RANK='0', WORLD_SIZE='1', LOCAL_RANK='0',
                     PYTHONPATH=f'{ROOT}:{ROOT / "third_party/geniedrive_occ_render"}')
    worker_env = os.environ.copy()
    worker_env.update(overrides)
    metadata['environment_overrides'] = overrides
    command = [sys.executable, '-u', str(ROOT / 'tools/genie/inference/infer.py'),
               '--genie-dir', str(ROOT / 'third_party/geniedrive_occ_render'),
               '--base-model', str(args.base_model), '--checkpoint', str(args.checkpoint),
               '--data-dir', str(conditions_dir), '--meta-path', str(meta_path),
               '--save-dir', str(output), '--height', str(args.height), '--width', str(args.width),
               '--video-length', str(frame_count), '--fps', str(fps),
               '--num-inference-steps', str(args.num_inference_steps),
               '--guidance-scale', str(args.guidance_scale), '--seed', str(args.seed),
               '--gpu-memory-mode', args.gpu_memory_mode, '--dtype', 'bf16',
               '--max-samples', '1', '--save-grid']
    if args.overwrite:
        command.append('--overwrite')
    metadata['command'] = command
    log_path = output / 'logs' / 'inference.log'
    write_json(output / 'manifest.json', metadata)
    logger.info('Generate %d frames at %d FPS on GPU %s; output: %s', frame_count, fps, args.gpu, output)
    logger.info('Start inference worker; full log: %s', log_path)
    try:
        run_logged_process(command, cwd=ROOT, log_path=log_path, env=worker_env)
        logger.info('Inference worker finished; checking six-camera outputs and condition sources')
        missing = [str(path) for path in expected if not path.is_file() or path.stat().st_size == 0]
        if missing:
            raise RuntimeError(f'Inference did not produce all six camera videos: {missing}')
        if (sha256_file(condition_manifest) != metadata['condition_manifest_sha256']
                or sha256_file(meta_path) != metadata['condition_meta_sha256']
                or any(file_identity(item['path']) != item for item in inputs)):
            raise RuntimeError('Condition inputs changed during video generation')
    except subprocess.CalledProcessError as error:
        # Surface the worker's actual exception instead of repeating its entire
        # launch command as the only diagnostic at the controller boundary.
        prefixes = ('RuntimeError:', 'ValueError:', 'FileNotFoundError:',
                    'ImportError:', 'ModuleNotFoundError:', 'torch.OutOfMemoryError:')
        reason = next((line for line in reversed(log_path.read_text().splitlines())
                       if line.startswith(prefixes)), f'Video worker exited with code {error.returncode}')
        metadata.update(status='failed', error=reason, error_type='WorkerError')
        raise RuntimeError(f'{reason} Full log: {log_path}') from None
    except BaseException as error:
        metadata.update(status='failed', error=str(error), error_type=type(error).__name__)
        raise
    else:
        metadata.update(status='completed', outputs=[path.name for path in expected],
                        grid='000000_grid.mp4')
    finally:
        metadata['finished_at'] = utc_now()
        write_json(output / 'manifest.json', metadata)
        lines = ['# 六相机生成视频', '', f"状态：**{metadata['status']}**。", '',
                 f'共 {frame_count} 帧，{fps} FPS，{args.width}×{args.height}。', '',
                 '- `000000_CAM_*.mp4`：六路生成视频。',
                 '- `000000_grid.mp4`：六路拼图视频。',
                 '- [推理日志](logs/inference.log)',
                 '- `logs/progress_rank0.json`：当前阶段、子阶段和耗时（worker 初始化后生成）。',
                 '- [来源、参数和状态](manifest.json)', '',
                 f'输入条件：`{conditions_dir}`', '',
                 '只有 completed 状态表示本次生成成功；失败时可能存在部分或此前的输出。', '']
        if metadata.get('error'):
            lines.extend([f"失败原因：{metadata['error']}", ''])
        with atomic_file(output / 'README.md', 'w') as stream:
            stream.write('\n'.join(lines))
    logger.info('Saved six-camera videos: %s', output)
    logger.info('Build generated surround and OCC/LiDAR overview GIFs')
    from tools.visualization.overview import render_overview
    render_overview(args.scene_dir)


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
                        datefmt='%H:%M:%S')
    run_scene(parse_args())


if __name__ == '__main__':
    main()
