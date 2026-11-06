#!/usr/bin/env python3
"""Benchmark the configured final network: joint encode, generation and joint decode."""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmengine.utils import import_modules_from_strings
from tools.bevworld_generation.runtime import configure_checkpoint_inference


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--warmup-samples', type=int, default=20)
    parser.add_argument('--measure-samples', type=int, default=200)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output')
    return parser.parse_args()


def build_model_and_dataset(args):
    cfg = Config.fromfile(args.config)
    if cfg.get('custom_imports'):
        import_modules_from_strings(**cfg.custom_imports)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))

    configure_checkpoint_inference(cfg, args.checkpoint)
    from mmdet3d.registry import DATASETS, MODELS
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu', strict=False)
    model.to(args.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    loader_cfg = cfg.get('test_dataloader') or cfg.get('val_dataloader')
    if loader_cfg is None:
        raise ValueError('Config requires a test_dataloader or val_dataloader.')
    dataset_cfg = loader_cfg.dataset
    dataset = DATASETS.build(dataset_cfg)
    return model, dataset


def prepare_generation_inputs(model, batch):
    data = model.data_preprocessor(batch, training=False)
    batch_inputs = data['inputs']
    data_samples = data['data_samples']
    device = batch_inputs['voxel_semantics'].device
    curr2futu_ego = torch.stack([
        torch.from_numpy(sample.metainfo['curr2futu_ego']).to(
            device).float()
        for sample in data_samples
    ])
    start_of_sequence = [
        sample.metainfo['start_of_sequence'] for sample in data_samples]
    current_index = model.prev_number
    encoder_inputs = {
        'voxel_semantics': batch_inputs['voxel_semantics'][:, current_index],
        'range_imgs': batch_inputs['range_imgs'][:, current_index],
    }
    return (batch_inputs, data_samples, encoder_inputs, start_of_sequence,
            curr2futu_ego)


@torch.inference_mode()
def benchmark_step(model, dataset, warmup_samples, measure_samples, seed):
    """Time the complete configured network without architecture overrides."""
    device = next(model.parameters()).device
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.memo_bev_latent = None

    requested = warmup_samples + measure_samples
    available = min(len(dataset), requested)
    if available <= warmup_samples:
        raise RuntimeError(
            f'Dataset has only {available} samples, not enough for '
            f'{warmup_samples} warm-up samples.')

    stage_seconds = {
        'encode': [],
        'generation': [],
        'decode': [],
        'end_to_end': [],
    }
    generated_frames = 0
    torch.cuda.reset_peak_memory_stats(device)

    for index in range(available):
        batch = pseudo_collate([dataset[index]])
        (batch_inputs, data_samples, encoder_inputs, start_of_sequence,
         curr2futu_ego) = prepare_generation_inputs(model, batch)
        torch.cuda.synchronize(device)
        end_to_end_start = time.perf_counter()

        encode_start = time.perf_counter()
        posterior, _, _ = model.vae.encode_fused(
            encoder_inputs, data_samples)
        curr_bev_latent = posterior.sample().unsqueeze(1)
        torch.cuda.synchronize(device)
        encode_elapsed = time.perf_counter() - encode_start

        memo_bev_latent = model.get_memo_latent(
            curr_bev_latent, start_of_sequence)
        generation_start = time.perf_counter()
        pred_futu_latent = model.rollout_future_latents(
            curr_bev_latent,
            memo_bev_latent,
            curr2futu_ego,
            bev_layout=batch_inputs.get('bev_layout'))
        torch.cuda.synchronize(device)
        generation_elapsed = time.perf_counter() - generation_start

        future_frames = pred_futu_latent.size(1)
        flat_futu_latent = pred_futu_latent.flatten(0, 1)
        decode_start = time.perf_counter()
        model.obtain_data_from_latent(
            flat_futu_latent,
            batch_data_samples=data_samples,
            frames_per_sample=future_frames)
        torch.cuda.synchronize(device)
        decode_elapsed = time.perf_counter() - decode_start
        end_to_end_elapsed = time.perf_counter() - end_to_end_start

        # Match BEVWorld.predict(): only observed latents update streaming
        # memory; generated future latents do not leak into the next sample.
        model.update_memo_latent(curr_bev_latent)
        if index >= warmup_samples:
            stage_seconds['encode'].append(encode_elapsed)
            stage_seconds['generation'].append(generation_elapsed)
            stage_seconds['decode'].append(decode_elapsed)
            stage_seconds['end_to_end'].append(end_to_end_elapsed)
            generated_frames += pred_futu_latent.size(0) * future_frames

    stage_arrays = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in stage_seconds.items()
    }
    measured_samples = len(stage_arrays['end_to_end'])
    stage_sum = (
        stage_arrays['encode']
        + stage_arrays['generation']
        + stage_arrays['decode'])
    end_to_end = stage_arrays['end_to_end']
    consistency_error = np.abs(stage_sum - end_to_end) / end_to_end

    def summarize(values):
        total_seconds = float(values.sum())
        return {
            'total_seconds': total_seconds,
            'milliseconds_per_sample_mean': float(1000.0 * values.mean()),
            'milliseconds_per_sample_std': float(1000.0 * values.std()),
            'milliseconds_per_sample_p50': float(
                1000.0 * np.percentile(values, 50)),
            'milliseconds_per_sample_p95': float(
                1000.0 * np.percentile(values, 95)),
        }

    stage_metrics = {
        name: summarize(values)
        for name, values in stage_arrays.items()
    }
    for name in ('generation', 'decode', 'end_to_end'):
        total_seconds = stage_metrics[name]['total_seconds']
        stage_metrics[name]['frames_per_second'] = float(
            generated_frames / total_seconds)
        stage_metrics[name]['milliseconds_per_frame'] = float(
            1000.0 * total_seconds / generated_frames)

    encode_generation_seconds = float(
        stage_arrays['encode'].sum() + stage_arrays['generation'].sum())
    fps = float(generated_frames / encode_generation_seconds)
    fps_e2e = float(generated_frames / float(stage_sum.sum()))

    return {
        'steps': int(model.flow_num_steps),
        'schedule': model.flow_inference_schedule,
        'warmup_samples': int(warmup_samples),
        'measured_samples': int(measured_samples),
        'generated_frames': int(generated_frames),
        'FPS': fps,
        'FPS-E2E': fps_e2e,
        'fps_definitions': {
            'FPS': 'generated future frames / measured (encode + generation) time',
            'FPS-E2E': 'generated future frames / measured (encode + generation + decode) time',
        },
        'stages': stage_metrics,
        'stage_sum_consistency_relative_error_mean': float(
            consistency_error.mean()),
        'stage_sum_consistency_relative_error_max': float(
            consistency_error.max()),
        'peak_memory_mib': float(
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)),
    }


def main():
    args = parse_args()
    if args.warmup_samples < 0 or args.measure_samples <= 0:
        raise ValueError('Warmup samples must be non-negative and measure samples positive.')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this FPS benchmark.')
    model, dataset = build_model_and_dataset(args)
    device = next(model.parameters()).device
    result = benchmark_step(
        model, dataset, args.warmup_samples, args.measure_samples, args.seed)
    report = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'device': str(device),
        'gpu': torch.cuda.get_device_name(device),
        'batch_size': 1,
        'context_block_count': len(model.context_blocks),
        'flow_block_count': len(model.flow_blocks),
        'future_frames_per_sample': result['generated_frames'] // result['measured_samples'],
        'timed_region': (
            'Tokenizer current-frame joint OCC/LiDAR encode + '
            'World future-latent generation + '
            'Tokenizer joint OCC/LiDAR decode of all generated frames. '
            'Data loading and preprocessing are excluded.'),
        'results': [result],
    }
    output = json.dumps(report, indent=2, ensure_ascii=False)
    print(output)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as file:
            file.write(output + '\n')


if __name__ == '__main__':
    main()
