#!/usr/bin/env python3
"""Model worker for scene generation; its process exits before rendering."""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import warnings
from pathlib import Path

# Keep known dependency deprecations from obscuring the scene-generation logs.
warnings.filterwarnings(
    "ignore", category=FutureWarning,
    message=r"`torch\.cuda\.amp\.custom_(fwd|bwd)\(args\.\.\.\)")
warnings.filterwarnings(
    "ignore", category=DeprecationWarning,
    message=r"^`TorchScript` support for functional optimizers is deprecated\b",
    module=r"^mmengine\.optim\.optimizer\.zero_optimizer$")

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[3]))

from tools.bevworld_generation.dataset import (
    current_latent_and_memo_inputs,
    prepare_batch,
    scene_start_index,
)
from tools.bevworld_generation.rollout import (
    build_layout_sequence,
    rollout,
)
from tools.bevworld_generation.runtime import build_runner
from tools.genie.occ_generation.generate_batch import (
    decode_occ_and_points,
)
from tools.genie.occ_generation.prepare_dataset import has_renderable_instances
from tools.genie.occ_generation.scene_timeline import (
    prepare_scene_timeline, timeline_prediction_fields, set_decode_frame_metadata,
)
from tools.precompute_nuscenes_lidar_range import (
    DEFAULT_PC_RANGE,
    _filter_egocar_points,
    _load_points,
    _resolve_lidar2ego,
    _resolve_lidar_path,
)
from projects.datasets.transforms.utils import pts2range_np
from tools.trajectory_generation.bev_layout import (
    current_to_future_transforms, render_layout_sequence,
    get_ego2global,
    get_info_location,
    get_lidar2ego,
    keyframe_info_to_render_info,
    load_scene_locations,
)
from tools.trajectory_generation.map_geometry_cache import SpatialBEVLayoutRenderer
from tools.genie.occ_generation.scene_runtime import atomic_file, file_identity, sha256_file
from tools.genie.occ_generation.scene_inputs import (
    load_scene_annotations, build_scene_dataset, cached_layout_arrays,
    warmup_scene_memo, input_cache_stats,
)
from tools.genie.occ_generation.scene_layouts import (
    CONTEXT_LAYOUT_CACHE_NAME, ensure_native_layouts,
)


DEFAULT_CONFIG = "configs/world_model/bev_world_12f_layout_autoregressive.py"
DEFAULT_CHECKPOINT = (
    "ckpts/bev_world_12f_layout_autoregressive.pth")
DEFAULT_ANN_12HZ = Path(
    "data/nus_infos_12hz/nuscenes_interp_12Hz_infos_val.pkl")
DEFAULT_ANN_2HZ = Path("data/nus_infos/nuscenes_infos_val.pkl")
LATENT_ROOT = Path("data/nus_occ_lidar_latent")
RANGE_ROOT = Path("data/nus_tokenizer_lidar_range")
DATA_ROOT = Path("data/nuscenes")
FUTURE_FRAMES = 80
DECODE_CHUNK_SIZE = 13
logger = logging.getLogger("generate_scene")


def render_future_layouts(frames, class_names):
    target_info, future_infos = frames[0], frames[1:]
    current_pose = get_ego2global(target_info)
    future_poses = np.stack([get_ego2global(info) for info in future_infos])
    current_to_future = current_to_future_transforms(current_pose, future_poses)

    data_root = DATA_ROOT
    scene_locations = load_scene_locations(str(data_root), "v1.0-trainval")
    renderer = SpatialBEVLayoutRenderer(str(data_root))
    render_infos = []
    for info in future_infos:
        if not has_renderable_instances(info):
            info = dict(info, instances=[])
        render_infos.append(keyframe_info_to_render_info(
            info, scene_locations, class_names, box_convention='legacy'))
    location = get_info_location(target_info, scene_locations)
    target_lidar2ego = get_lidar2ego(target_info)
    layouts = []
    for pose, render_info in tqdm(
            zip(future_poses, render_infos), total=len(render_infos),
            desc=f"Render {target_info['scene_name']} BEV"):
        layout = render_layout_sequence(
            renderer,
            pose[None],
            target_lidar2ego,
            [render_info],
            location,
        )
        layouts.append(layout[0])
    return np.stack(layouts, axis=0), current_to_future


def required_context_indices(dataset, target_index, warmup_frames=4):
    """Latents for warmup current frames plus the target sample history."""
    warmup_start = max(
        scene_start_index(dataset, target_index), target_index - warmup_frames)
    history = int(getattr(dataset, "load_prev_number", 0) or 0)
    history_start = max(scene_start_index(dataset, target_index), target_index - history)
    return list(range(min(warmup_start, history_start), target_index + 1))


def patch_bev_layout_roots(dataset, layout_root):
    """Point only this inference dataset at the scene-local layout cache."""
    transforms = getattr(dataset.pipeline, "transforms", [])
    for transform in transforms:
        if getattr(transform, "output_keys", None) == "bev_layout":
            transform.data_root = str(layout_root)
            transform.load_data_from_file.data_root = str(layout_root)
            return
    raise RuntimeError("The inference pipeline has no BEVLayout loader")


def ensure_bev_layouts(native_infos, class_names, tokens, cache_root):
    """Render native context geometry; validate provenance before cache reuse."""
    result = ensure_native_layouts(
        SpatialBEVLayoutRenderer(str(DATA_ROOT)), native_infos, class_names, tokens,
        cache_root, load_scene_locations(str(DATA_ROOT), 'v1.0-trainval'), DATA_ROOT)
    logger.info('Native context layouts: %s; cache=%s', result, cache_root)
    return result


def ensure_range_image(info, token):
    """Generate the tokenizer's range image from its raw LiDAR file."""
    path = RANGE_ROOT / f"{token}.npz"
    if path.is_file():
        return path
    logger.info("Range image missing for %s; generating from raw LiDAR", token)
    lidar_path = _resolve_lidar_path(info, DATA_ROOT)
    points = _load_points(lidar_path, load_dim=5, use_dim=[0, 1, 2, 3, 4])
    points = _filter_egocar_points(points, ego_width=1.85, ego_length=4.084)
    range_img = pts2range_np(
        points,
        lidar2ego=_resolve_lidar2ego(info),
        H=32,
        W=1024,
        min_depth=1.45,
        max_depth=80.0,
        fov_up=10.0,
        fov_down=-30.0,
        pc_range=DEFAULT_PC_RANGE,
        pc_range_coord="ego",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    range_imgs = range_img[None].astype(np.float32, copy=False)
    with atomic_file(path) as stream:
        np.savez_compressed(
            stream, range_imgs=range_imgs,
            points_coord_after_transform="lidar", lidar_path=str(lidar_path),
            pc_range=np.asarray(DEFAULT_PC_RANGE, dtype=np.float32),
            pc_range_coord="ego",
            range_image_shape=np.asarray(range_imgs.shape, dtype=np.int32))
    logger.info("Saved generated range image: %s", path)
    return path


def ensure_latent(model, info, token):
    """Encode a missing OCC+LiDAR tokenizer latent with the loaded VAE."""
    path = LATENT_ROOT / f"{token}.npz"
    if path.is_file():
        return path
    logger.info("OCC/LiDAR latent missing for %s; encoding with the VAE", token)
    range_path = ensure_range_image(info, token)
    occ_path = Path(info.get("occ3d_path", ""))
    if not occ_path.is_file():
        occ_path = DATA_ROOT / 'gts' / str(info["scene_name"]) / token / "labels.npz"
    if not occ_path.is_file():
        raise FileNotFoundError(
            f"Cannot create latent for {token}: OCC label is missing at {occ_path}")
    with np.load(occ_path) as archive:
        if "semantics" in archive:
            semantics = archive["semantics"]
        elif "voxel_label" in archive:
            semantics = archive["voxel_label"]
        else:
            raise KeyError(f"No semantics array in {occ_path}")
    with np.load(range_path) as archive:
        range_imgs = archive["range_imgs"]

    from mmdet3d.structures import Det3DDataSample

    lidar2ego = _resolve_lidar2ego(info)
    data_sample = Det3DDataSample(metainfo=dict(
        token=token, lidar2ego=lidar2ego,
        ego2lidar=np.linalg.inv(lidar2ego)))
    device = next(model.vae.parameters()).device
    inputs = dict(
        voxel_semantics=torch.as_tensor(
            semantics[None], dtype=torch.long, device=device),
        range_imgs=torch.as_tensor(
            range_imgs[None], dtype=torch.float32, device=device))
    data_sample = data_sample.to(device)
    model.vae.eval()
    with torch.no_grad():
        posterior, _, _ = model.vae.encode_fused(inputs, [data_sample])
        seed = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "little")
        devices = [device.index] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            latent = posterior.sample()
    LATENT_ROOT.mkdir(parents=True, exist_ok=True)
    with atomic_file(path) as stream:
        np.savez_compressed(
            stream,
            latent=latent[0].detach().cpu().numpy(),
            posterior_mean=posterior.mean[0].detach().cpu().numpy())
    logger.info("Saved generated latent: %s", path)
    return path


def ensure_context_inputs(model, dataset, indices, native_infos, class_names,
                          layout_cache):
    native_by_token = {str(info["token"]): info for info in native_infos}
    tokens = [str(dataset.get_data_info(index)["token"]) for index in indices]
    missing_latents = sum(
        not (LATENT_ROOT / f"{token}.npz").is_file() for token in tokens)
    logger.info(
        "Checking %d context frames (%d missing latents)", len(tokens), missing_latents)
    layout_indices = required_context_indices(dataset, indices[-1], 0)
    layout_tokens = [str(dataset.get_data_info(i)["token"]) for i in layout_indices]
    ensure_bev_layouts(native_infos, class_names, layout_tokens, layout_cache)
    patch_bev_layout_roots(dataset, layout_cache)
    for token in tqdm(tokens, desc="Check context OCC/LiDAR latents", unit="frame"):
        info = native_by_token.get(token)
        if info is None:
            raise KeyError(
                f"No native 2 Hz nuScenes info found for context token {token}")
        if not (LATENT_ROOT / f"{token}.npz").is_file():
            ensure_latent(model, info, token)
    logger.info("Context inputs are ready for %d frames", len(tokens))


def select_scene_window(annotations, valid_tokens, future_frames, start_token=None):
    infos = annotations['infos']
    for start, info in enumerate(infos):
        if start_token is not None and str(info['token']) != start_token:
            continue
        frames = infos[start:start + future_frames + 1]
        if len(frames) != future_frames + 1:
            if start_token is not None:
                raise ValueError(f'Start token {start_token} has fewer than {future_frames} future frames')
            break
        same_scene = all(frame['scene_name'] == info['scene_name'] for frame in frames)
        if (same_scene and str(info['token']) in valid_tokens
                and bool(info.get('is_key_frame', True))):
            return frames
        if start_token is not None:
            raise ValueError(f'Start token {start_token} is not an eligible native keyframe in this scene')
    if start_token is not None:
        raise ValueError(f'Start token {start_token} was not found in this scene')
    raise ValueError(f'No native keyframe with {future_frames} future frames in this scene')


def layout_signature(frames, class_names, data_root):
    """Cover actual conditions, map source, geometry and renderer implementation."""
    import pickle

    locations = load_scene_locations(str(data_root), 'v1.0-trainval')
    location = get_info_location(frames[0], locations)
    map_path = Path(data_root) / 'maps' / 'expansion' / f'{location}.json'
    sources = [Path(__file__), Path('tools/trajectory_generation/bev_layout.py'),
               Path('tools/trajectory_generation/map_geometry_cache.py'),
               Path('tools/trajectory_generation/box_conventions.py')]
    return dict(
        version=1, frame_inputs_sha256=hashlib.sha256(
            pickle.dumps((frames, tuple(class_names)), protocol=4)).hexdigest(),
        data_root=str(Path(data_root).resolve()), location=location,
        map=file_identity(map_path), grid=dict(origin=[-40, -40, -1], voxel_size=0.4,
                                             layout_shape=[18, 200, 200]),
        coordinate='future ego pose with current-frame lidar extrinsic',
        implementation={str(p): sha256_file(p) for p in sources})


def run_inference(args, report):
    """Produce the existing prediction protocol in a dedicated model process."""
    global FUTURE_FRAMES, LATENT_ROOT, RANGE_ROOT, DATA_ROOT
    from mmengine.config import Config
    from mmengine.utils import import_modules_from_strings
    from mmengine.registry import init_default_scope

    FUTURE_FRAMES = args.future_frames
    args.cfg_options = None
    args.dataset_split = 'val'
    with report.stage('configuration'):
        cfg = Config.fromfile(args.config)
        if cfg.get('custom_imports'):
            import_modules_from_strings(**cfg.custom_imports)
        init_default_scope(cfg.get('default_scope', 'mmdet3d'))
        native_path = Path(cfg.val_dataloader.dataset.ann_file)
        LATENT_ROOT = Path(cfg.occ_range_latent_root)
        RANGE_ROOT = Path(cfg.occ_range_target_root)
        DATA_ROOT = Path(cfg.data_root)
        Path(args.out_dir, 'metadata', 'resolved_config.py').write_text(cfg.pretty_text)
        report.data['source_config_sha256'] = hashlib.sha256(cfg.pretty_text.encode()).hexdigest()
    cache_root = args._cache_dir or Path(args.out_dir) / '.cache'
    scene_name = f'scene-{int(args.scene):04d}'
    with report.stage('scene_annotations'):
        future, future_hit = load_scene_annotations(
            args.ann_12hz, scene_name, cache_root, refresh=args.refresh_cache)
        candidates = [str(info['token']) for i, info in enumerate(future['infos'])
                      if i + FUTURE_FRAMES < len(future['infos'])
                      and bool(info.get('is_key_frame', True))]
        start_token = getattr(args, 'start_token', None)
        if start_token is not None:
            if start_token not in candidates:
                raise ValueError(f'Start token {start_token} is not eligible for this scene/horizon')
            candidates = [start_token]
        native, native_hit = load_scene_annotations(
            native_path, scene_name, cache_root,
            load_interval=cfg.val_dataloader.dataset.get('load_interval', 1),
            refresh=args.refresh_cache, candidate_tokens=candidates)
        valid_tokens = {str(info['token']) for info in native['infos']}
        frames = select_scene_window(future, valid_tokens, FUTURE_FRAMES, start_token)
        target_token = str(frames[0]['token'])
        local_indices = {str(info['token']): i for i, info in enumerate(native['infos'])}
        target_index = local_indices[target_token]
        context_layout_root = cache_root / CONTEXT_LAYOUT_CACHE_NAME
        dataset = build_scene_dataset(cfg, native, context_layout_root)
        report.data.update(
            scene=scene_name, native_scene=native['source_scene'], target_token=target_token,
            source_dataset_index=native['source_indices'][target_index],
            scene_dataset_index=target_index, scene_native_frames=len(dataset),
            annotation_cache_hits=dict(native=native_hit, future=future_hit),
            context_layout_cache=str(context_layout_root),
            annotation_sources=dict(native=native['signature'], future=future['signature']))
    with report.stage('timeline'):
        frames, timeline_metadata = prepare_scene_timeline(
            args, frames, native, future, cfg.data_root, args.out_dir)
        report.data['timeline'] = timeline_metadata
    with report.stage('future_layouts'):
        trajectory_file = getattr(args, 'trajectory_file', None)
        trajectory_fields = {}
        if trajectory_file:
            from tools.genie.occ_generation.scene_trajectories import (
                candidate_decode_frames, load_trajectory_condition)
            layouts, current_to_future, trajectory = load_trajectory_condition(
                trajectory_file, args.trajectory_index, frames)
            report.data['trajectory'] = trajectory
            frames = candidate_decode_frames(frames)
            trajectory_fields = dict(
                trajectory_name=np.asarray(trajectory['name']),
                trajectory_index=np.asarray(args.trajectory_index),
                trajectory_source_sha256=np.asarray(trajectory['source']['sha256']))
            logger.info('Trajectory candidate %d: %s', args.trajectory_index, trajectory['name'])
        else:
            signature = layout_signature(frames, future['class_names'], cfg.data_root)
            layouts, current_to_future, hit = cached_layout_arrays(
                cache_root / 'future_layouts.npz', signature,
                lambda: render_future_layouts(frames, future['class_names']),
                FUTURE_FRAMES, refresh=args.refresh_cache)
            report.data['future_layout_cache_hit'] = hit
            report.data['future_layout_signature'] = signature
            logger.info('Future layout cache: %s', 'hit' if hit else 'miss')
    with report.stage('provenance'):
        report.data['checkpoint'] = dict(**file_identity(args.checkpoint),
                                         sha256=sha256_file(args.checkpoint))
        report.data['source_sha256'] = {
            str(p): sha256_file(p) for p in
            [Path(__file__), Path('tools/genie/occ_generation/scene_inputs.py'),
             Path('tools/genie/occ_generation/scene_layouts.py'),
             Path('tools/genie/occ_generation/scene_trajectories.py'),
             Path('tools/trajectory_generation/box_conventions.py'),
             Path('tools/genie/occ_generation/generate_batch.py')]}
    with report.stage('load_model'):
        args.runner_work_dir = str(Path(args.out_dir) / 'logs' / 'mmengine')
        runner = build_runner(args, quiet_startup=True)
        model = runner.model
        # The runner clears training-only initializer checkpoints and sets
        # load_from/launcher; persist that effective inference configuration.
        effective_config = runner.cfg.pretty_text
        Path(args.out_dir, 'metadata', 'resolved_config.py').write_text(effective_config)
        report.data['resolved_config_sha256'] = hashlib.sha256(effective_config.encode()).hexdigest()
    device = next(model.parameters()).device
    synchronize = lambda: torch.cuda.synchronize(device)
    with report.stage('context_inputs', synchronize):
        indices = required_context_indices(dataset, target_index, args.warmup_frames)
        ensure_context_inputs(model, dataset, indices, native['infos'],
                              native['class_names'], context_layout_root)
    with report.stage('memory_warmup', synchronize):
        warmed = warmup_scene_memo(model, dataset, target_index, args.warmup_frames)
    with report.stage('prepare_current', synchronize):
        data = prepare_batch(model, dataset, target_index)
        inputs, samples, current_latent, start_sequence = current_latent_and_memo_inputs(model, data)
        set_decode_frame_metadata(samples, frames, model)
        memo_latent = model.get_memo_latent(current_latent, start_sequence)
        layout_sequence = build_layout_sequence(model, inputs['bev_layout'], layouts)
        ego = torch.as_tensor(current_to_future, device=device, dtype=torch.float32)[None]
    model_input_peak = torch.cuda.max_memory_allocated(device) / 2**20
    with report.stage('rollout', synchronize):
        with tqdm(total=FUTURE_FRAMES, desc='Generate future latents', unit='frame') as progress:
            future_latent, rollout_seconds, rollout_peak = rollout(
                model, current_latent, memo_latent, ego, layout_sequence,
                FUTURE_FRAMES, args.seed, progress_callback=progress.update)
    with report.stage('decode', synchronize):
        all_latents = torch.cat([current_latent, future_latent], dim=1)
        all_occ, all_points = decode_occ_and_points(
            model, all_latents, samples, args.decode_chunk_size,
            progress_desc='Decode OCC and LiDAR', start_frame=model.prev_number)
    with report.stage('save_predictions'):
        point_sequence = [np.asarray(p[:, :3], dtype=np.float32) for p in all_points[1:]]
        if len(all_occ) != FUTURE_FRAMES + 1 or len(all_points) != FUTURE_FRAMES + 1:
            raise ValueError('Decoded sequence does not match the requested horizon')
        offsets = np.zeros(FUTURE_FRAMES + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(p) for p in point_sequence])
        packed = np.concatenate(point_sequence, axis=0)
        tokens = np.asarray([str(info['token']) for info in frames[1:]])
        current_transform = samples[0].metainfo.get('lidar2ego')
        if current_transform is None:
            current_transform = np.asarray(samples[0].metainfo['total_lidar2ego'])[model.prev_number]
        prediction_path = Path(args.out_dir) / 'predictions' / 'sequence.npz'
        with atomic_file(prediction_path) as stream:
            np.savez_compressed(
                stream, target_token=np.asarray(target_token), future_tokens_12hz=tokens,
                current_occ=all_occ[0], pred_occ_12hz=all_occ[1:],
                current_points_xyz=np.asarray(all_points[0][:, :3], dtype=np.float32),
                pred_points_xyz_concat=packed, pred_points_xyz_offsets=offsets,
                point_coordinate=np.asarray('lidar'),
                point_lidar2ego=np.asarray(current_transform, dtype=np.float32),
                current_bev_layout=inputs['bev_layout'][0, model.prev_number].detach().cpu().numpy().astype(np.uint8),
                pred_points_lidar2ego=np.stack([get_lidar2ego(f) for f in frames[1:]]).astype(np.float32),
                bev_layout=layouts, candidate_curr2futu_ego=current_to_future.astype(np.float32),
                scene_name=np.asarray(scene_name), generation_seed=np.asarray(args.seed),
                **timeline_prediction_fields(frames, timeline_metadata),
                **trajectory_fields)
    report.data.update(
        seed=args.seed, warmup_tokens=warmed, future_frames=FUTURE_FRAMES,
        input_array_cache=input_cache_stats(dataset), rollout_seconds=rollout_seconds,
        rollout_peak_allocated_mib=rollout_peak,
        inference_peak_allocated_mib=max(model_input_peak, torch.cuda.max_memory_allocated(device) / 2**20),
        prediction_sha256=sha256_file(prediction_path))
    report.flush()
    # The worker process terminates here, releasing runner hooks and CUDA state.
