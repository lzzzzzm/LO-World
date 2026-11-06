"""Freeze the physical clock actually consumed by a scene generation run."""
from pathlib import Path
import pickle

import numpy as np

from tools.genie.occ_generation.scene_runtime import atomic_file, sha256_file
from tools.trajectory_generation.resample_scene_timeline import (
    DEFAULT_TIMELINE_CACHE, TIMELINE_VERSION, resample_frames, validate_timeline, write_timeline,
)


def validate_trajectory_clock(path):
    """Reject old sweep-clock archives before any new scene generation."""
    with np.load(path, allow_pickle=False) as data:
        tokens = np.asarray(data['future_tokens']).astype(str)
        if tokens.ndim != 1 or not len(tokens) or not all(t.startswith('uniform12_') for t in tokens):
            raise ValueError('Old 12 Hz trajectory clock is unsupported; rebuild the trajectory '
                             'and its Layout from uniform resampled annotations.')
        if 'timeline_version' in data and str(data['timeline_version']) != TIMELINE_VERSION:
            raise ValueError('Trajectory timeline version does not match uniform 12 Hz')


def prepare_scene_timeline(args, frames, native, source, data_root, run_dir):
    if getattr(args, 'trajectory_file', None):
        validate_trajectory_clock(args.trajectory_file)
    destination = Path(run_dir) / 'inputs/timeline'
    if frames[1].get('timeline_version') != TIMELINE_VERSION:
        from tools.trajectory_generation.raw_timeline_index import RawTimelineIndex
        index = RawTimelineIndex(data_root, getattr(args, 'timeline_cache_dir', DEFAULT_TIMELINE_CACHE))
        try:
            start = frames[0]
            native_start = next(f for f in native['infos'] if str(f['token']) == str(start['token']))
            frames, timeline, details = resample_frames(
                start, native_start, native['class_names'], index.scene(str(start['token'])),
                data_root, len(frames)-1)
        finally:
            index.close()
        write_timeline(destination, frames, source['metainfo'], timeline, details)
    validate_timeline(frames)
    # Freeze annotations so render-only runs remain portable with their prediction.
    annotation = destination / 'infos.pkl'
    with atomic_file(annotation) as stream:
        pickle.dump(dict(data_list=frames, metainfo=source['metainfo'],
                         timeline_version=TIMELINE_VERSION),
                    stream, protocol=pickle.HIGHEST_PROTOCOL)
    metadata = dict(mode='uniform', version=TIMELINE_VERSION,
                    fps=12, annotation='inputs/timeline/infos.pkl', annotation_sha256=sha256_file(annotation),
                    source_annotation=source['signature']['source'])
    return frames, metadata


def timeline_prediction_fields(frames, metadata):
    return dict(timeline_version=np.asarray(metadata['version']),
                frame_timestamps_us=np.asarray([f['timestamp'] for f in frames]),
                timeline_annotation=np.asarray('../inputs/timeline/infos.pkl'),
                timeline_annotation_sha256=np.asarray(metadata['annotation_sha256']))


def set_decode_frame_metadata(data_samples, frames, model):
    """Attach the exact current/future sensor clock to joint decoding."""
    from tools.trajectory_generation.bev_layout import get_lidar2ego
    if not data_samples or not frames:
        raise ValueError('Cannot attach decode transforms without samples and frames')
    transforms = np.stack([get_lidar2ego(frame) for frame in frames])
    tokens = [str(frame['token']) for frame in frames]
    prefix_count = int(model.prev_number)
    prefix = np.repeat(transforms[:1], prefix_count, axis=0)
    data_samples[0].set_metainfo(dict(
        total_lidar2ego=np.concatenate([prefix, transforms]),
        total_token=[tokens[0]] * prefix_count + tokens))
