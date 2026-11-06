"""Explicit box encoding boundaries; all rigid frame transforms use native boxes."""

import numpy as np


BOX_TRANSFORM_VERSION = 'native_boxes_before_rigid_transform_v1'


def convert_boxes(boxes, source, target):
    """Convert native [l,w,h,yaw] <-> legacy [w,l,h,-yaw-pi/2].

    Legacy yaw is an encoding for a renderer with a local reflection, not a
    physical direction vector. Decode it before applying an SE(3) transform.
    """
    if source not in ('native', 'legacy') or target not in ('native', 'legacy'):
        raise ValueError(f'Unknown box convention: {source} -> {target}')
    values = np.asarray(boxes, dtype=np.float64)
    if values.size == 0:
        return values.reshape(0, 7).copy()
    if values.ndim != 2 or values.shape[1] != 7 or not np.isfinite(values).all():
        raise ValueError('Boxes must be a finite [N,7] array')
    values = values.copy()
    if source != target:
        values[:, [3, 4]] = values[:, [4, 3]]
        values[:, 6] = -values[:, 6] - np.pi / 2
    return values


def transform_boxes_between_lidars(boxes, source_lidar2global,
                                   target_lidar2global):
    """Transform native box centers and headings between LiDAR frames."""
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.size == 0:
        return boxes.reshape(0, 7)
    source_to_target = np.linalg.inv(target_lidar2global) @ source_lidar2global
    centers_hom = np.concatenate([
        boxes[:, :3], np.ones((len(boxes), 1), dtype=np.float64)
    ], axis=1)
    transformed = boxes.copy()
    transformed[:, :3] = (source_to_target @ centers_hom.T).T[:, :3]
    forward = np.stack([
        np.cos(boxes[:, 6]), np.sin(boxes[:, 6]),
        np.zeros(len(boxes), dtype=np.float64)
    ], axis=1)
    target_forward = (source_to_target[:3, :3] @ forward.T).T
    transformed[:, 6] = np.arctan2(
        target_forward[:, 1], target_forward[:, 0])
    return transformed
