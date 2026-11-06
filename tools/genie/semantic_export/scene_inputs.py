"""Resolve semantic-render inputs from a generated scene directory."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from tools.genie.occ_generation.scene_outputs import resolve_prediction
from tools.genie.occ_generation.scene_runtime import sha256_file


def prediction_annotation(prediction):
    """Resolve and authenticate a prediction's own frozen annotation."""
    prediction = Path(prediction)
    with np.load(prediction, allow_pickle=False) as archive:
        annotation = prediction.parent / str(archive['timeline_annotation'])
        expected = str(archive['timeline_annotation_sha256'])
    if not annotation.is_file() or sha256_file(annotation) != expected:
        raise ValueError(f'Frozen prediction timeline is missing or changed: {annotation}')
    return annotation.resolve()


def validate_prediction_clock(prediction, frames, fps=12):
    from tools.trajectory_generation.resample_scene_timeline import TIMELINE_VERSION, validate_timeline
    if fps != 12:
        raise ValueError('Export FPS must match the 12 Hz physical timeline')
    with np.load(prediction, allow_pickle=False) as archive:
        if str(archive['timeline_version']) != TIMELINE_VERSION:
            raise ValueError('Prediction must use the uniform physical 12 Hz timeline')
        expected = archive['frame_timestamps_us'][:len(frames)]
        actual = np.asarray([f['timestamp'] for f in frames])
        if not np.array_equal(expected, actual):
            raise ValueError('Prediction and render annotation timestamps disagree')
        validate_timeline(frames)


def scene_defaults(scene_dir: Path) -> dict:
    """Freeze the successful prediction path before rendering, never follow latest again."""
    scene_dir = scene_dir.resolve()
    prediction = resolve_prediction(scene_dir)
    selected = json.loads((prediction.parent.parent / 'manifest.json').read_text())
    return dict(
        predictions=prediction,
        out_dir=scene_dir / 'conditions',
        source_run_id=selected['run_id'],
        prediction_sha256=selected['prediction']['sha256'],
    )
