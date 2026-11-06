"""Validate saved trajectory and Layout conditions for scene generation.

Trajectory and layout must be changed together. Tokens identify the original
scene timeline; the candidate poses describe the counterfactual ego motion.
"""

from pathlib import Path

import numpy as np

from tools.genie.occ_generation.scene_runtime import file_identity, sha256_file
from tools.trajectory_generation.bev_layout import get_ego2global, get_lidar2ego
from tools.trajectory_generation.box_conventions import BOX_TRANSFORM_VERSION
from tools.trajectory_generation.resample_scene_timeline import validate_timeline
from tools.genie.occ_generation.scene_timeline import validate_trajectory_clock


def validate_rigid_transforms(values, shape, name):
    values = np.asarray(values)
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError(f'{name} must contain finite transforms with shape {shape}')
    rotation = values[..., :3, :3]
    if (not np.allclose(values[..., 3, :], [0, 0, 0, 1], atol=1e-5, rtol=0)
            or not np.allclose(rotation.swapaxes(-1, -2) @ rotation,
                               np.eye(3), atol=1e-4, rtol=0)
            or not np.allclose(np.linalg.det(rotation), 1, atol=1e-4, rtol=0)):
        raise ValueError(f'{name} must contain rigid, right-handed transforms')
    return values


def load_trajectory_condition(path, index, frames):
    """Load one candidate, rejecting conditions for a different start or clock."""
    path = Path(path)
    validate_timeline(frames)
    validate_trajectory_clock(path)
    source = dict(**file_identity(path), sha256=sha256_file(path))
    count = len(frames) - 1
    if count < 1:
        raise ValueError('Trajectory conditions need at least one future frame')
    with np.load(path, allow_pickle=False) as archive:
        if (int(archive['schema_version']) < 4
                or 'box_transform_version' not in archive
                or str(archive['box_transform_version']) != BOX_TRANSFORM_VERSION
                or str(archive['dynamic_yaw_convention']) != 'physical_frame_transform'):
            raise ValueError('Trajectory conditions must be regenerated with schema >= 4 '
                             'and native boxes before rigid transforms; old conditions may have wrong headings')
        if (str(archive['scene_name']) != str(frames[0]['scene_name'])
                or str(archive['target_token']) != str(frames[0]['token'])):
            raise ValueError('Trajectory scene/target token does not match this scene window')
        expected_tokens = np.asarray([str(f['token']) for f in frames[1:]])
        if (int(archive['num_frames']) != count
                or not np.array_equal(archive['future_tokens'], expected_tokens)
                or archive['future_padding_mask'].shape != (count,)
                or archive['future_padding_mask'].any()):
            raise ValueError('Trajectory horizon/timeline must match the unpadded 12 Hz scene window')
        if ('frame_timestamps_us' in archive and not np.array_equal(
                archive['frame_timestamps_us'], [f['timestamp'] for f in frames])):
            raise ValueError('Trajectory timestamps do not match the uniform scene window')
        if ('future_timestamps_us' in archive and not np.array_equal(
                archive['future_timestamps_us'], [f['timestamp'] for f in frames[1:]])):
            raise ValueError('Trajectory future timestamps do not match the uniform scene window')
        names = archive['trajectory_names']
        if names.ndim != 1 or not 0 <= index < len(names):
            raise ValueError(f'Trajectory index {index} is outside the candidate list')
        total = len(names)
        current = validate_rigid_transforms(archive['current_ego2global'], (4, 4), 'current_ego2global')
        calibration = validate_rigid_transforms(archive['lidar2ego'], (4, 4), 'lidar2ego')
        if (not np.allclose(current, get_ego2global(frames[0]), atol=2e-3, rtol=0)
                or not np.allclose(calibration, get_lidar2ego(frames[0]), atol=1e-5, rtol=0)):
            raise ValueError('Trajectory initial pose or LiDAR calibration does not match the current frame')
        poses = archive['candidate_ego2global']
        transforms = archive['candidate_curr2futu_ego']
        layouts = archive['bev_layout']
        if (poses.shape != (total, count, 4, 4)
                or transforms.shape != (total, count, 4, 4)
                or layouts.shape != (total, count, 18, 200, 200)):
            raise ValueError('Candidate pose/transform/layout dimensions do not match the timeline')
        poses = validate_rigid_transforms(poses[index], (count, 4, 4), 'candidate_ego2global')
        transforms = validate_rigid_transforms(
            transforms[index], (count, 4, 4), 'candidate_curr2futu_ego')
        # Exported global poses are float32; metre-scale comparisons allow 2 mm
        # for cancellation when world coordinates are hundreds of metres large.
        expected = np.linalg.inv(poses.astype(np.float64)) @ current.astype(np.float64)
        if not np.allclose(transforms, expected, atol=2e-3, rtol=0):
            raise ValueError('Trajectory ego transforms disagree with candidate global poses')
        layout = layouts[index]
        if layout.dtype != np.uint8 or np.any(layout > 1):
            raise ValueError('Trajectory BEV layouts must be binary uint8 masks')
        metadata = dict(source=source, index=index, name=str(names[index]),
                        box_transform_version=str(archive['box_transform_version']),
                        dynamic_yaw_convention=str(archive['dynamic_yaw_convention']))
        for key in ('path_length', 'final_yaw_deg', 'drivable_ratio', 'min_vehicle_clearance_m'):
            if key in archive:
                metadata[key] = float(archive[key][index])
    if file_identity(path) != {k: v for k, v in source.items() if k != 'sha256'}:
        raise RuntimeError('Trajectory condition file changed while it was being read')
    return layout, transforms, metadata


def candidate_decode_frames(frames):
    """Candidates use the target LiDAR rig at every predicted ego pose."""
    calibration = get_lidar2ego(frames[0])
    return [frames[0]] + [dict(
        frame, lidar_points=dict(frame.get('lidar_points', {}), lidar2ego=calibration))
        for frame in frames[1:]]
