#!/usr/bin/env python3
"""Create final Waymo 2Hz infos and TOP dual-return point clouds.

Run from the repository root. Requires original MMDetection3D infos, TFRecords
and Occ3D labels. Sample IDs/timestamps follow the frozen final split manifest;
clouds contain TOP-frame float32 XYZ/intensity/elongation. The model pipeline
performs the final 64x2048 projection, not this converter.
"""

import argparse
import json
import multiprocessing as mp
import os
import pickle
from pathlib import Path

import numpy as np


_TIMESTAMP_TO_SAMPLE_ID = None
_TIMESTAMP_TO_POSE = None
_OCC_ROOT = None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['train', 'val', 'all'], default='all')
    parser.add_argument('--data-root', type=Path, default=Path('data/waymo'))
    parser.add_argument('--train-ann', type=Path,
                        help='Original training infos; defaults to DATA_ROOT/waymo_info/train_10hz.pkl')
    parser.add_argument('--val-ann', type=Path,
                        help='Original validation infos; defaults to DATA_ROOT/waymo_info/waymo_infos_val.pkl')
    parser.add_argument('--val-occ-root', type=Path,
                        help='Validation occupancy labels; defaults to DATA_ROOT/validation')
    parser.add_argument('--splits-file', type=Path,
                        default=Path(__file__).resolve().parent / 'waymo' / 'splits.json',
                        help='Fixed split manifest; defaults to waymo/splits.json beside this script')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--annotations-only', action='store_true')
    mode.add_argument('--points-only', action='store_true',
                      help='Use existing fixed 2Hz annotation without reopening original infos')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max-records', type=int, help='Debug only: limit TFRecords')
    parser.add_argument('--max-frames', type=int, help='Debug only: limit original frames per record')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def create_waymo_infos(source, split, output_path):
    """Select the final 2Hz infos by frozen sample ID and timestamp."""
    if output_path.exists():
        with output_path.open('rb') as stream:
            existing = pickle.load(stream)
        assert [[int(x['image']['image_idx']), int(x['timestamp'])]
                for x in existing] == split['frames']
        print('Existing fixed selection verified:', output_path, len(existing))
        return
    if source is None:
        raise FileNotFoundError(f'Missing fixed annotation: {output_path}')
    with Path(source).open('rb') as stream:
        infos = pickle.load(stream)
    by_id = {int(info['image']['image_idx']): info for info in infos}
    assert len(by_id) == len(infos), 'Duplicate annotation sample IDs'
    selected = []
    for sid, timestamp in split['frames']:
        info = by_id[sid]
        assert int(info['timestamp']) == timestamp, sid
        selected.append(info)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('wb') as stream:
        pickle.dump(selected, stream)
    print('Created fixed selection:', output_path, len(selected))


def _load_timestamp_map(ann_file):
    with open(ann_file, 'rb') as file:
        annotations = pickle.load(file)
    mapping = {}
    for info in annotations:
        timestamp = int(info['timestamp'])
        sample_id = int(info['image']['image_idx'])
        if timestamp in mapping and mapping[timestamp] != sample_id:
            raise ValueError(f'duplicate Waymo timestamp: {timestamp}')
        mapping[timestamp] = sample_id
    return mapping


def _init_worker(ann_file, occ_root):
    global _TIMESTAMP_TO_SAMPLE_ID, _TIMESTAMP_TO_POSE, _OCC_ROOT
    global tf, dataset_pb2, range_image_utils, transform_utils
    global parse_range_image_and_camera_projection
    import tensorflow as tf
    from waymo_open_dataset import dataset_pb2
    from waymo_open_dataset.utils import range_image_utils, transform_utils
    from waymo_open_dataset.utils.frame_utils import parse_range_image_and_camera_projection
    _OCC_ROOT = Path(occ_root)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    _TIMESTAMP_TO_SAMPLE_ID = _load_timestamp_map(ann_file)
    with open(ann_file, 'rb') as f:
        _TIMESTAMP_TO_POSE = {int(x['timestamp']): np.asarray(x['pose']) for x in pickle.load(f)}


def _range_return_to_features(frame, range_images, top_pose, return_index):
    calibrations = sorted(
        frame.context.laser_calibrations, key=lambda calibration: calibration.name)
    frame_pose = tf.convert_to_tensor(
        np.asarray(frame.pose.transform).reshape(4, 4))
    top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(top_pose.data), top_pose.shape.dims)
    rotation = transform_utils.get_rotation_matrix(
        top_pose_tensor[..., 0],
        top_pose_tensor[..., 1],
        top_pose_tensor[..., 2])
    top_pose_transform = transform_utils.get_transform(
        rotation, top_pose_tensor[..., 3:])

    features = []
    for calibration in calibrations:
        if calibration.name != dataset_pb2.LaserName.TOP:
            continue
        range_image = range_images[calibration.name][return_index]
        range_tensor = tf.reshape(
            tf.convert_to_tensor(range_image.data), range_image.shape.dims)
        valid = (range_tensor[..., 0] > 0) & (range_tensor[..., 3] != 1.0)
        indices = tf.where(valid)

        if len(calibration.beam_inclinations) == 0:
            inclinations = range_image_utils.compute_inclination(
                tf.constant([
                    calibration.beam_inclination_min,
                    calibration.beam_inclination_max,
                ]),
                height=range_image.shape.dims[0])
        else:
            inclinations = tf.constant(calibration.beam_inclinations)
        inclinations = tf.reverse(inclinations, axis=[-1])
        extrinsic = np.asarray(
            calibration.extrinsic.transform).reshape(4, 4)

        pixel_pose = None
        vehicle_pose = None
        if calibration.name == dataset_pb2.LaserName.TOP:
            pixel_pose = tf.expand_dims(top_pose_transform, axis=0)
            vehicle_pose = tf.expand_dims(frame_pose, axis=0)
        cartesian = range_image_utils.extract_point_cloud_from_range_image(
            tf.expand_dims(range_tensor[..., 0], axis=0),
            tf.expand_dims(extrinsic, axis=0),
            tf.expand_dims(inclinations, axis=0),
            pixel_pose=pixel_pose,
            frame_pose=vehicle_pose)
        xyz = tf.gather_nd(tf.squeeze(cartesian, axis=0), indices)
        intensity = tf.gather_nd(range_tensor[..., 1], indices)
        elongation = tf.gather_nd(range_tensor[..., 2], indices)
        features.append(tf.concat([
            xyz,
            intensity[:, None],
            elongation[:, None],
        ], axis=1).numpy())
    if not features:
        raise ValueError('selected Waymo frame contains no matching LiDAR')
    return np.concatenate(features, axis=0)


def _frame_to_points(frame):
    parsed = parse_range_image_and_camera_projection(frame)
    range_images, top_pose = parsed[0], parsed[-1]
    if top_pose is None:
        raise ValueError('Waymo frame has no TOP LiDAR pose')
    returns = [
        _range_return_to_features(frame, range_images, top_pose, index)
        for index in (0, 1)
    ]
    return np.concatenate(returns, axis=0).astype(np.float32, copy=False)


def _top_lidar2ego(frame):
    top_calibration = next(
        calibration for calibration in frame.context.laser_calibrations
        if calibration.name == dataset_pb2.LaserName.TOP)
    return np.asarray(
        top_calibration.extrinsic.transform, dtype=np.float64).reshape(4, 4)


def _ego_points_to_top(points, top_lidar2ego):
    xyz_hom = np.concatenate([
        points[:, :3].astype(np.float64, copy=False),
        np.ones((points.shape[0], 1), dtype=np.float64),
    ], axis=1)
    converted = points.copy()
    converted[:, :3] = (
        np.linalg.inv(top_lidar2ego) @ xyz_hom.T).T[:, :3]
    return converted.astype(np.float32, copy=False)


def _convert_record(task):
    record_path, output_dir, max_frames, overwrite = task
    output_dir = Path(output_dir)
    converted = skipped = 0
    sample_ids = []
    context_name = None
    top_lidar2ego = None
    dataset = tf.data.TFRecordDataset(record_path, compression_type='')
    for frame_index, serialized in enumerate(dataset):
        if max_frames is not None and frame_index >= max_frames:
            break
        if frame_index % 5:
            continue
        frame = dataset_pb2.Frame()
        frame.ParseFromString(serialized.numpy())
        context_name = frame.context.name
        timestamp = int(frame.timestamp_micros)
        if timestamp not in _TIMESTAMP_TO_SAMPLE_ID:
            raise KeyError(
                f'{Path(record_path).name}: timestamp {timestamp} is absent '
                'from the annotation')
        sample_id = _TIMESTAMP_TO_SAMPLE_ID[timestamp]
        sample_ids.append(sample_id)
        raw_pose = np.asarray(frame.pose.transform).reshape(4, 4)
        assert np.allclose(raw_pose, _TIMESTAMP_TO_POSE[timestamp], rtol=0, atol=1e-5), sample_id
        scene, index = (sample_id // 1000) % 1000, sample_id % 1000
        occ_path = _OCC_ROOT / f'{scene:03d}' / f'{index:03d}_04.npz'
        with np.load(occ_path) as occ:
            assert occ['voxel_label'].shape == (200, 200, 16), occ_path
            assert set(np.unique(occ['voxel_label']).tolist()) <= set(range(15)) | {23}, occ_path
            assert np.allclose(occ['ego2global'], raw_pose, rtol=0, atol=1e-5), occ_path
        output_path = output_dir / f'{sample_id:07d}.bin'
        frame_top_lidar2ego = _top_lidar2ego(frame)
        if top_lidar2ego is None:
            top_lidar2ego = frame_top_lidar2ego
        elif not np.allclose(top_lidar2ego, frame_top_lidar2ego, atol=1e-6):
            raise ValueError(
                f'{Path(record_path).name}: TOP extrinsic changes within one TFRecord')
        if (not overwrite and output_path.is_file() and
                output_path.stat().st_size > 0 and
                output_path.stat().st_size % (5 * 4) == 0):
            skipped += 1
            continue
        points = _frame_to_points(frame)
        points = _ego_points_to_top(points, top_lidar2ego)
        temporary_path = output_path.with_suffix('.bin.tmp')
        points.tofile(temporary_path)
        os.replace(temporary_path, output_path)
        converted += 1
    return {
        'record': Path(record_path).name,
        'context_name': context_name,
        'converted': converted,
        'skipped': skipped,
        'frames': converted + skipped,
        'sample_id_min': min(sample_ids) if sample_ids else None,
        'sample_id_max': max(sample_ids) if sample_ids else None,
        'sample_ids': sample_ids,
        'top_lidar2ego': (top_lidar2ego.tolist()
                          if top_lidar2ego is not None else None),
    }


def convert_waymo_points(args, input_dir, ann_file, occ_root, output_dir, manifest_path):
    if args.workers < 1:
        raise ValueError('--workers must be positive')
    records = sorted(Path(input_dir).glob('*.tfrecord'))
    if args.max_records is not None:
        records = records[:args.max_records]
    if not records:
        raise FileNotFoundError(f'no TFRecords found under {input_dir}')

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (str(record), str(output_dir), args.max_frames, args.overwrite)
        for record in records
    ]
    context = mp.get_context('spawn')
    if args.workers == 1:
        _init_worker(ann_file, occ_root)
        results = [_convert_record(task) for task in tasks]
    else:
        with context.Pool(
                args.workers,
                initializer=_init_worker,
                initargs=(ann_file, occ_root)) as pool:
            results = list(pool.imap_unordered(_convert_record, tasks))

    results.sort(key=lambda item: item['record'])
    manifest = {
        'input_dir': os.path.relpath(input_dir),
        'ann_file': os.path.relpath(ann_file),
        'output_dir': os.path.relpath(output_dir),
        'records': len(results),
        'frames': sum(item['frames'] for item in results),
        'converted': sum(item['converted'] for item in results),
        'skipped': sum(item['skipped'] for item in results),
        'sensor_selection': ['TOP'],
        'returns': [1, 2],
        'output_frame': 'top',
        'results': results,
    }
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w', encoding='utf-8') as file:
        json.dump(manifest, file, indent=2)
        file.write('\n')
    print(json.dumps({
        key: manifest[key]
        for key in ('records', 'frames', 'converted', 'skipped')
    }))


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError('--workers must be positive')
    for limit in [args.max_records, args.max_frames]:
        if limit is not None and limit < 1:
            raise ValueError('Debug limits must be positive')
    fixed = json.loads(args.splits_file.read_text())
    splits = ['train', 'val'] if args.split == 'all' else [args.split]
    for label in splits:
        info_dir = args.data_root / 'waymo_info'
        ann_file = info_dir / f'{label}_2hz.pkl'
        if not args.points_only:
            source = (args.train_ann or info_dir / 'train_10hz.pkl'
                      ) if label == 'train' else (args.val_ann or info_dir / 'waymo_infos_val.pkl')
            create_waymo_infos(source, fixed['splits'][label], ann_file)
        else:
            # Check the selected split even when annotation creation is skipped.
            create_waymo_infos(None, fixed['splits'][label], ann_file)
        if args.annotations_only:
            continue
        convert_waymo_points(
            args, args.data_root / 'raw' / label, ann_file,
            args.data_root / 'training' if label == 'train'
            else (args.val_occ_root or args.data_root / 'validation'),
            args.data_root / 'points' / label,
            info_dir / f'{label}_conversion_manifest.json')


if __name__ == '__main__':
    main()
