"""Point-cloud conversion and serialization helpers."""

from __future__ import annotations

import numpy as np


def tensor_to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def points_to_numpy(points):
    return tensor_to_numpy(points[:, :min(4, points.shape[1])])


def pack_point_sequence(points_sequence, default_dim=4):
    """Pack variable-length point frames without object dtype or pickle."""
    if points_sequence is None:
        return None
    frames = []
    point_dim = default_dim
    for points in points_sequence:
        points = np.asarray(points, dtype=np.float32)
        if points.ndim == 1:
            points = points.reshape(0, point_dim) if points.size == 0 else points.reshape(1, -1)
        if points.ndim != 2:
            raise ValueError(
                f"Expected point frame with shape [N, C], got {points.shape}.")
        if points.shape[0] > 0:
            point_dim = max(point_dim, points.shape[1])
        frames.append(points)
    offsets = np.zeros(len(frames) + 1, dtype=np.int64)
    for index, points in enumerate(frames):
        offsets[index + 1] = offsets[index] + points.shape[0]
    concat = np.zeros((int(offsets[-1]), point_dim), dtype=np.float32)
    cursor = 0
    for points in frames:
        next_cursor = cursor + points.shape[0]
        concat[cursor:next_cursor, :points.shape[1]] = points
        cursor = next_cursor
    return {
        "concat": concat,
        "offsets": offsets,
        "dim": np.asarray(point_dim, dtype=np.int64),
    }
