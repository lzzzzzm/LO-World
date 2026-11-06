#!/usr/bin/env python3
"""Generate one 12 Hz nuScenes scene with isolated inference and rendering."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import subprocess
import sys
import time

SCRIPT_STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.genie.occ_generation.scene_runtime import StageTimings, file_identity
from tools.genie.occ_generation.scene_outputs import (
    SceneBundle, resolve_prediction, run_logged_process,
)

DEFAULT_CONFIG = ROOT / 'configs/world_model/bev_world_12f_layout_autoregressive.py'
DEFAULT_CHECKPOINT = (
    ROOT / 'ckpts/bev_world_12f_layout_autoregressive.pth')
DEFAULT_ANN_12HZ = ROOT / 'data/nus_infos_12hz/nuscenes_interp_12Hz_infos_val.pkl'
logger = logging.getLogger('generate_scene')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', required=True, help='Four-digit scene code, e.g. 0001')
    parser.add_argument('--start-token',
                        help='Use this native keyframe token instead of the first eligible scene frame')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--render-only', action='store_true',
                        help='Render the latest successful saved prediction')
    parser.add_argument('--predictions', type=Path,
                        help='Explicit prediction NPZ to use with --render-only')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--ann-12hz', type=Path, default=DEFAULT_ANN_12HZ)
    parser.add_argument('--timeline-cache-dir', type=Path, default=ROOT / 'work_dirs/scene_timeline_cache',
                        help='Shared raw geometry index; reused across scenes and batch preparation')
    parser.add_argument('--future-frames', type=int, default=80)
    parser.add_argument('--warmup-frames', type=int, default=4)
    parser.add_argument('--decode-chunk-size', type=int, default=13)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--trajectory-file', type=Path,
                        help='Saved trajectory/Layout NPZ (12 Hz, matching scene window)')
    parser.add_argument('--trajectory-index', type=int, default=0,
                        help='Candidate index in --trajectory-file; default 0 is usually the GT baseline')
    parser.add_argument('--refresh-cache', action='store_true',
                        help='Rebuild scene annotations and future BEV layouts')
    parser.add_argument('--no-render', action='store_true', help='Save predictions without a GIF')
    parser.add_argument('--export-point-frames', action='store_true',
                        help='Export current and future LiDAR frames in addition to the packed NPZ')
    parser.add_argument('--keep-render-frames', action='store_true',
                        help='Keep full-resolution PNGs under visualization/frames/')
    parser.add_argument('--_inference-worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--_run-id', help=argparse.SUPPRESS)
    parser.add_argument('--_cache-dir', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not args.scene.isdigit():
        parser.error('--scene must contain digits only')
    if args.future_frames < 1 or args.decode_chunk_size < 1 or args.warmup_frames < 0:
        parser.error('Frame count/chunk size must be positive and warmup nonnegative')
    if not 0 <= args.seed < 2**32:
        parser.error('--seed must be in [0, 2**32)')
    if args.render_only and args.no_render:
        parser.error('--render-only and --no-render cannot be combined')
    if args.predictions and not args.render_only:
        parser.error('--predictions requires --render-only')
    if args.trajectory_index < 0 or (args.trajectory_index and not args.trajectory_file):
        parser.error('--trajectory-index must be nonnegative and requires --trajectory-file')
    if args.trajectory_file and args.render_only:
        parser.error('--trajectory-file cannot be combined with --render-only')
    if args.start_token and args.render_only:
        parser.error('--start-token cannot be combined with --render-only')
    for key in ('out_dir', 'config', 'checkpoint', 'ann_12hz', 'timeline_cache_dir'):
        setattr(args, key, getattr(args, key).resolve())
    for key in ('predictions', '_cache_dir', 'trajectory_file'):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())
    args.scene = f'{int(args.scene):04d}'
    return args


def inference_worker(args):
    report = StageTimings(args.out_dir / 'metadata' / 'inference.json',
                          dict(scene=f'scene-{args.scene}', seed=args.seed, run_id=args._run_id))
    try:
        # Fail before loading annotations, maps, model registrations or weights.
        with report.stage('cuda_preflight'):
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('Scene generation requires an exposed CUDA GPU; '
                                   'use --render-only for existing predictions')
            report.data['gpu'] = dict(name=torch.cuda.get_device_name(),
                                     visible_devices=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'))
        with report.stage('imports'):
            from tools.genie.occ_generation.scene_inference import run_inference
        run_inference(args, report)
    except BaseException as error:
        import traceback
        report.data['traceback'] = traceback.format_exc()
        report.finish(error)
        raise
    else:
        report.finish()


def save_3d_visualization(predictions_path, output_path, *, keep_frames=False,
                          log_path=None, run_id=None):
    """Run only after the model worker has exited, so no model holds GPU memory."""
    predictions_path, output_path = Path(predictions_path), Path(output_path)
    command = [sys.executable, str(ROOT / 'tools/visualization/scene.py'),
               '--predictions', str(predictions_path), '--output', str(output_path)]
    metadata_path = output_path.parent / 'render_meta.json'
    command.extend(['--metadata', str(metadata_path),
                    '--frames-dir', str(output_path.parent / 'frames')])
    if run_id:
        command.extend(['--run-id', run_id])
    if keep_frames:
        command.append('--keep-frames')
    run_logged_process(command, cwd=ROOT, log_path=log_path or output_path.parent / 'render.log')
    return json.loads(metadata_path.read_text())


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
                        datefmt='%H:%M:%S')
    args = parse_args(argv)
    if args._inference_worker:
        inference_worker(args)
        return
    parameters = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                  if not k.startswith('_')}
    with SceneBundle(args.out_dir, f'scene-{args.scene}',
                     'render_only' if args.render_only else 'generate', parameters) as bundle:
        execute_scene(args, bundle, argv)
    logger.info('Result bundle: %s', bundle.path)


def execute_scene(args, bundle, argv):
    report = StageTimings(bundle.path / 'metadata' / 'run.json', dict(
        mode=bundle.mode, run_id=bundle.run_id, scene=bundle.scene,
        command=[sys.executable, str(Path(__file__).resolve()),
                 *(sys.argv[1:] if argv is None else argv)],
        parameters=bundle.parameters, process_isolation=True))
    bundle.artifact('运行记录', report.path, '本次整体状态、命令和阶段耗时')
    if not args.render_only:
        bundle.data['generation'] = dict(config=str(args.config), checkpoint=str(args.checkpoint),
                                          seed=args.seed)
    try:
        with report.stage('preflight'):
            prediction_path = ((args.predictions or resolve_prediction(bundle.root)) if args.render_only
                               else bundle.path / 'predictions' / 'sequence.npz')
            sources = [prediction_path] if args.render_only else [args.config, args.checkpoint, args.ann_12hz]
            if args.trajectory_file:
                sources.append(args.trajectory_file)
            report.data['inputs'] = [file_identity(path) for path in sources]
        if not args.render_only:
            with report.stage('inference_process'):
                # Reuse the parsed options, independent of how main() was invoked.
                command = [sys.executable, str(Path(__file__).resolve()), '--_inference-worker',
                           '--scene', args.scene, '--out-dir', str(bundle.path),
                           '--_run-id', bundle.run_id, '--_cache-dir', str(bundle.cache)]
                for option in ('config', 'checkpoint', 'ann_12hz', 'future_frames',
                               'warmup_frames', 'decode_chunk_size', 'seed', 'timeline_cache_dir'):
                    command.extend(['--' + option.replace('_', '-'), str(getattr(args, option))])
                for option in ('refresh_cache',):
                    if getattr(args, option):
                        command.append('--' + option.replace('_', '-'))
                if args.start_token:
                    command.extend(['--start-token', args.start_token])
                if args.trajectory_file:
                    command.extend(['--trajectory-file', str(args.trajectory_file),
                                    '--trajectory-index', str(args.trajectory_index)])
                try:
                    run_logged_process(command, cwd=ROOT,
                                       log_path=bundle.path / 'logs' / 'inference.log')
                except subprocess.CalledProcessError as error:
                    metadata_path = bundle.path / 'metadata' / 'inference.json'
                    try:
                        metadata = json.loads(metadata_path.read_text())
                        # A worker can exit before writing a report. Do not
                        # attribute an older run's failure to the current run.
                        current = (isinstance(metadata, dict)
                                   and metadata.get('run_id') == bundle.run_id)
                    except (OSError, ValueError):
                        metadata, current = {}, False
                    if current and metadata.get('error'):
                        report.data['inference'] = metadata
                        stage = metadata.get('failed_stage', 'inference')
                        reason = str(metadata['error']).splitlines()[0]
                        kind = metadata.get('error_type', 'Error')
                        raise RuntimeError(
                            f'Inference worker failed during {stage}: {kind}: {reason}. '
                            f'Full details: {metadata_path}') from None
                    raise RuntimeError(
                        f'Inference worker exited with code {error.returncode} '
                        'without a current failure report; see its traceback above.') from None
            report.data['inference'] = json.loads((bundle.path / 'metadata' / 'inference.json').read_text())
            bundle.data['generation']['checkpoint_sha256'] = report.data['inference']['checkpoint']['sha256']
            if args.trajectory_file:
                bundle.data['generation']['trajectory'] = report.data['inference']['trajectory']
        with report.stage('document_predictions'):
            if args.render_only:
                prediction_path = bundle.attach_prediction(prediction_path)
            bundle.describe_prediction(prediction_path, export_points=args.export_point_frames)
        if not args.no_render:
            with report.stage('render_process'):
                rendering = save_3d_visualization(
                    prediction_path, bundle.path / 'visualization' / 'overview.gif',
                    keep_frames=args.keep_render_frames,
                    log_path=bundle.path / 'logs' / 'render.log', run_id=bundle.run_id)
                if rendering['source_sha256'] != bundle.data['prediction']['sha256']:
                    raise RuntimeError('Prediction file changed while rendering; result is not published')
            report.data['rendering'] = dict(animation=rendering.get('animation'),
                                            timings=rendering.get('stages_seconds'),
                                            keep_frames=args.keep_render_frames)
            bundle.artifact('OCC 与 LiDAR 动画', bundle.path / 'visualization' / 'overview.gif',
                            '白底对照动画；第 0 帧为当前重建，其余为未来预测')
            bundle.artifact('渲染参数', bundle.path / 'visualization' / 'render_meta.json',
                            '配色、视角、帧数、耗时和预测文件 SHA256')
    except BaseException as error:
        report.data['total_process_seconds'] = time.perf_counter() - SCRIPT_STARTED
        report.finish(error)
        raise
    else:
        report.data['total_process_seconds'] = time.perf_counter() - SCRIPT_STARTED
        report.finish()
        logger.info('Scene task complete in %.2f s; timing/provenance: %s',
                    report.data['total_process_seconds'], report.path)
    finally:
        for name, relative, description in (
                ('推理记录', 'metadata/inference.json', '模型来源、阶段耗时及失败堆栈'),
                ('实际配置', 'metadata/resolved_config.py', 'Runner 使用的配置'),
                ('推理日志', 'logs/inference.log', '推理子进程完整输出'),
                ('渲染日志', 'logs/render.log', '渲染子进程完整输出')):
            path = bundle.path / relative
            if path.exists():
                bundle.artifact(name, path, description)


if __name__ == '__main__':
    main()
