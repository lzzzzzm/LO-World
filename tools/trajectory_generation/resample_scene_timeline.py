"""Physical-time resampling of raw nuScenes poses and instance tracks."""
from __future__ import annotations

from collections import defaultdict
import copy
import json
from pathlib import Path
import pickle

import numpy as np
from pyquaternion import Quaternion

from tools.genie.occ_generation.scene_runtime import atomic_file, sha256_file, write_json
from tools.trajectory_generation.box_conventions import convert_boxes


def iter_json_array(path):
    decoder = json.JSONDecoder()
    buffer, pos, eof = '', 0, False
    with Path(path).open() as stream:
        while True:
            if not eof:
                chunk = stream.read(1 << 20)
                eof = not chunk
                buffer, pos = buffer[pos:] + chunk, 0
            while pos < len(buffer):
                if buffer[pos] in ' \r\n\t[,':
                    pos += 1
                    continue
                if buffer[pos] == ']':
                    return
                try:
                    record, end = decoder.raw_decode(buffer, pos)
                except json.JSONDecodeError:
                    if eof:
                        raise
                    break
                yield record
                pos = end
            if eof:
                raise ValueError(f'Incomplete JSON array: {path}')


def pose_matrix(translation, rotation):
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Quaternion(rotation).rotation_matrix
    result[:3, 3] = translation
    return result


def bracket(times, timestamp):
    times = np.asarray(times)
    if timestamp < times[0] or timestamp > times[-1]:
        raise ValueError(f'Time {timestamp} outside [{times[0]}, {times[-1]}]; extrapolation forbidden')
    upper = int(np.searchsorted(times, timestamp, side='left'))
    if times[upper] == timestamp:
        return upper, upper, 0.
    lower = upper - 1
    return lower, upper, float((timestamp - times[lower]) / (times[upper] - times[lower]))


def interpolate_pose(records, timestamp):
    a, b, alpha = bracket([r['timestamp'] for r in records], timestamp)
    old, new = records[a], records[b]
    translation = (1 - alpha) * np.asarray(old['translation']) + alpha * np.asarray(new['translation'])
    rotation = Quaternion.slerp(Quaternion(old['rotation']), Quaternion(new['rotation']), alpha)
    return pose_matrix(translation, rotation.elements)


def world_box_to_lidar(center, size_wlh, rotation, lidar2global):
    inverse = np.linalg.inv(lidar2global)
    local_center = inverse[:3, :3] @ np.asarray(center) + inverse[:3, 3]
    # Match the native nuScenes converter's Box.orientation.yaw_pitch_roll
    # convention, including the small roll/pitch contribution of the rig.
    local_rotation = Quaternion(matrix=lidar2global[:3, :3]).inverse * Quaternion(rotation)
    yaw = local_rotation.yaw_pitch_roll[0]
    return np.array([*local_center, size_wlh[1], size_wlh[0], size_wlh[2], yaw])


def tracks_from_raw(raw, class_names):
    from nuscenes.eval.detection.utils import category_to_detection_name
    samples = {r['token']: r for r in raw['samples']}
    tracks = defaultdict(list)
    for ann in raw['annotations']:
        category_token = raw['instances'][ann['instance_token']]['category_token']
        name = category_to_detection_name(raw['category'][category_token]['name'])
        if name not in class_names:
            continue
        tracks[ann['instance_token']].append(dict(ann, timestamp=samples[ann['sample_token']]['timestamp'], name=name))
    return {key: sorted(values, key=lambda a: a['timestamp']) for key, values in tracks.items()}


def interpolate_tracks(tracks, timestamp, lidar2global, class_names):
    instances = []
    for instance_id, records in sorted(tracks.items()):
        if timestamp < records[0]['timestamp'] or timestamp > records[-1]['timestamp']:
            continue
        a, b, alpha = bracket([r['timestamp'] for r in records], timestamp)
        old, new = records[a], records[b]
        # Match the native valid-point policy; within an interval retain a
        # track if either endpoint has observed points. Never extrapolate it.
        if not any(r['num_lidar_pts'] + r['num_radar_pts'] > 0 for r in (old, new)):
            continue
        center = (1-alpha)*np.asarray(old['translation']) + alpha*np.asarray(new['translation'])
        size = (1-alpha)*np.asarray(old['size']) + alpha*np.asarray(new['size'])
        rotation = Quaternion.slerp(Quaternion(old['rotation']), Quaternion(new['rotation']), alpha)
        native_box = world_box_to_lidar(center, size, rotation.elements, lidar2global)
        legacy_box = convert_boxes(native_box[None], 'native', 'legacy')[0]
        label = class_names.index(new['name'])
        instances.append(dict(instance_token=instance_id, bbox_3d=legacy_box.tolist(),
                              bbox_label=label, bbox_label_3d=label, bbox_3d_isvalid=True,
                              num_lidar_pts=int(new['num_lidar_pts']), num_radar_pts=int(new['num_radar_pts']),
                              world_center=center.tolist(), world_size_wlh=size.tolist(),
                              world_rotation_wxyz=rotation.elements.tolist(),
                              source_annotation_tokens=[old['token'], new['token']], interpolation_alpha=alpha))
    return instances


def validate_current(rebuilt, source, class_names):
    from scipy.optimize import linear_sum_assignment
    old = [r for r in source['instances'] if r.get('bbox_3d_isvalid', True)
           and int(r.get('bbox_label_3d', -1)) in range(len(class_names))]
    report = dict(raw_valid_boxes=len(rebuilt), native_valid_boxes=len(old), max_center_error_m=0.,
                  max_size_error_m=0., max_yaw_error_rad=0.)
    assert len(old) == len(rebuilt), report
    for label in range(len(class_names)):
        before = np.array([r['bbox_3d'] for r in old if r['bbox_label_3d'] == label]).reshape(-1, 7)
        after = np.array([r['bbox_3d'] for r in rebuilt if r['bbox_label_3d'] == label]).reshape(-1, 7)
        after = convert_boxes(after, 'legacy', 'native')
        assert len(before) == len(after), (class_names[label], len(before), len(after))
        if not len(before):
            continue
        costs = np.linalg.norm(before[:, None, :3] - after[None, :, :3], axis=-1)
        rows, cols = linear_sum_assignment(costs)
        errors = np.abs(before[rows] - after[cols])
        yaw_error = np.abs(np.arctan2(np.sin(before[rows, 6]-after[cols, 6]), np.cos(before[rows, 6]-after[cols, 6])))
        report['max_center_error_m'] = max(report['max_center_error_m'], float(costs[rows, cols].max()))
        report['max_size_error_m'] = max(report['max_size_error_m'], float(errors[:, 3:6].max()))
        report['max_yaw_error_rad'] = max(report['max_yaw_error_rad'], float(yaw_error.max()))
    assert report['max_center_error_m'] < 1e-5, report
    assert report['max_size_error_m'] < 1e-6, report
    assert report['max_yaw_error_rad'] < 1e-5, report
    return report


TIMELINE_VERSION = 'raw_pose_instance_tracks_uniform_v1'
DEFAULT_TIMELINE_CACHE = Path('work_dirs/scene_timeline_cache')


def resample_frames(start, native_start, class_names, raw, data_root,
                    future_frames=80, fps=12):
    """One shared physical-time implementation for scene and batch consumers."""
    if fps < 1 or future_frames < 1:
        raise ValueError('FPS and future frame count must be positive')
    data_root = Path(data_root)
    class_names = tuple(class_names)
    tracks = tracks_from_raw(raw, class_names)
    by_channel = defaultdict(list)
    for record in raw['sensor_data']:
        calibration = raw['calibrated_sensor'][record['calibrated_sensor_token']]
        channel = raw['sensor'][calibration['sensor_token']]['channel']
        by_channel[channel].append(record)
    for values in by_channel.values():
        values.sort(key=lambda r: r['timestamp'])
    lidar_records = [raw['ego'][r['ego_pose_token']] for r in by_channel['LIDAR_TOP']]
    t0 = int(start['timestamp'])
    timestamps = t0 + np.rint(np.arange(future_frames+1) * 1e6 / fps).astype(np.int64)
    poses = np.stack([interpolate_pose(lidar_records, int(t)) for t in timestamps])
    assert np.allclose(poses[0], start['ego2global'], atol=1e-9, rtol=0)
    lidar2ego = np.asarray(start['lidar_points']['lidar2ego'])
    # The preserved native current frame has authoritative annotations. A
    # track can have a missing annotation at this sample even when observed
    # both before and after it; interpolating that gap is not a native oracle.
    current_tracks = {key: [record for record in records
                           if record['sample_token'] == str(start['token'])]
                      for key, records in tracks.items()}
    current_tracks = {key: records for key, records in current_tracks.items() if records}
    rebuilt_start = interpolate_tracks(current_tracks, t0, poses[0] @ lidar2ego, class_names)
    current_check = validate_current(rebuilt_start, native_start, class_names)
    frames, counts = [], []
    for i, (timestamp, pose) in enumerate(zip(timestamps, poses)):
        frame = copy.deepcopy(start)
        if i == 0:
            # Preserve the exact existing current condition/reference images.
            frames.append(frame)
            counts.append(len(rebuilt_start))
            continue
        frame.update(token=f'uniform{fps}_{start["token"]}_{int(timestamp)}',
                     timestamp=int(timestamp), ego2global=pose.tolist(), is_key_frame=False,
                     is_scene_start=False, scene_token_index=i, sample_idx=i,
                     timeline_version=TIMELINE_VERSION, box_convention='legacy')
        frame['instances'] = interpolate_tracks(tracks, int(timestamp), pose @ lidar2ego, class_names)
        counts.append(len(frame['instances']))
        for camera, image in frame['images'].items():
            records = by_channel[camera]
            nearest = min(records, key=lambda r: abs(r['timestamp'] - int(timestamp)))
            image.update(img_path=str(data_root / nearest['filename']),
                         timestamp=int(nearest['timestamp']), sample_data_token=nearest['token'],
                         render_timestamp_us=int(timestamp),
                         source_to_target_time_error_us=int(nearest['timestamp'] - timestamp))
            # cam2ego/cam2img stay the same reference rig calibration. Geometry
            # is rendered in ego(t); the real RGB retains its acquisition time.
            image.pop('lidar2cam', None)
        # Synthetic future frames are conditions, never GT labels/point clouds.
        frame.pop('occ3d_path', None)
        frame['lidar_points'].pop('lidar_path', None)
        frames.append(frame)
    tokens = [str(f['token']) for f in frames]
    for frame in frames[1:]:
        frame['scene_clip_tokens'] = tokens
    timeline = dict(timestamps_us=timestamps, ego2global=poses,
                    curr2futu_ego=np.linalg.inv(poses[1:]) @ poses[0], lidar2ego=lidar2ego,
                    tokens=np.asarray(tokens), valid_instance_counts=np.asarray(counts))
    report = dict(schema=TIMELINE_VERSION, frames=len(frames), fps=fps,
                  dt_us=np.diff(timestamps).tolist(), current_validation=current_check,
                  tracks=len(tracks), instance_counts=counts, raw_scene_signature=raw['signature'],
                  birth_death_policy='inside observed track support only; no extrapolation',
                  validity_policy='either bounding annotation has sensor points; exact knots use that annotation')
    return frames, timeline, report


def write_timeline(out_dir, frames, metainfo, timeline, report, **provenance):
    out_dir = Path(out_dir)
    with atomic_file(out_dir / 'infos.pkl') as stream:
        annotation = dict(data_list=frames, metainfo=metainfo, timeline_version=TIMELINE_VERSION)
        pickle.dump(annotation, stream, protocol=pickle.HIGHEST_PROTOCOL)
    with atomic_file(out_dir / 'timeline.npz') as stream:
        np.savez_compressed(stream, **timeline)
    report = dict(report, status='completed', target_token=str(frames[0]['token']),
                  infos_sha256=sha256_file(out_dir / 'infos.pkl'),
                  timeline_sha256=sha256_file(out_dir / 'timeline.npz'), script_sha256=sha256_file(__file__),
                  **provenance)
    write_json(out_dir / 'report.json', report)
    return report


def validate_timeline(frames, fps=12):
    """A version label alone cannot establish timestamp or token consistency."""
    if len(frames) < 2:
        raise ValueError('A timeline needs a current and future frame')
    timestamps = np.asarray([f['timestamp'] for f in frames], dtype=np.int64)
    expected = timestamps[0] + np.rint(np.arange(len(frames))*1e6/fps).astype(np.int64)
    if not np.array_equal(timestamps, expected):
        raise ValueError('Uniform timeline timestamps do not match frame cadence')
    for frame in frames[1:]:
        token = f'uniform{fps}_{frames[0]["token"]}_{int(frame["timestamp"])}'
        if frame.get('timeline_version') != TIMELINE_VERSION or frame['token'] != token:
            raise ValueError('Uniform timeline frame version/token does not match its timestamp')
    return timestamps
