import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mmengine
import numpy as np

from projects.datasets.transforms.utils import pts2range_np


DEFAULT_PC_RANGE = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]


def _load_infos(ann_file):
    data = mmengine.load(ann_file)
    if isinstance(data, dict):
        if 'data_list' in data:
            return data['data_list']
        if 'infos' in data:
            return data['infos']
    if isinstance(data, list):
        return data
    raise TypeError(
        f'Unsupported annotation format in {ann_file}: {type(data)}')


def _resolve_lidar_path(info, data_root):
    lidar_points = info.get('lidar_points', {})
    lidar_path = lidar_points.get('lidar_path', info.get('lidar_path'))
    if lidar_path is None:
        raise KeyError(f'Missing lidar path for token {info.get("token")}.')

    path = Path(lidar_path)
    candidates = [path]
    if not path.is_absolute() and data_root is not None:
        candidates.append(Path(data_root) / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def _resolve_lidar2ego(info):
    lidar_points = info.get('lidar_points', {})
    lidar2ego = lidar_points.get('lidar2ego', info.get('lidar2ego'))
    if lidar2ego is None:
        raise KeyError(f'Missing lidar2ego for token {info.get("token")}.')
    lidar2ego = np.asarray(lidar2ego, dtype=np.float32)
    if lidar2ego.shape != (4, 4):
        raise ValueError(
            f'Expected lidar2ego shape (4, 4) for token {info.get("token")}, '
            f'got {lidar2ego.shape}.')
    return lidar2ego


def _load_points(path, load_dim, use_dim):
    if path.suffix == '.npy':
        points = np.load(path)
        if points.ndim == 1:
            points = points.reshape(-1, load_dim)
    else:
        points = np.fromfile(path, dtype=np.float32).reshape(-1, load_dim)
    return points[:, use_dim].astype(np.float32, copy=False)


def _filter_egocar_points(points, ego_width, ego_length):
    mask = np.logical_or(
        np.abs(points[:, 1]) > ego_width / 2,
        np.logical_or(points[:, 0] > ego_length / 2 + 1.0,
                      points[:, 0] < -ego_length / 2 + 0.5))
    return points[mask]


def convert_one(job):
    (token, lidar_path, dst_root, suffix, params, lidar2ego, load_dim, use_dim,
     filter_egocar_points, ego_width, ego_length, overwrite) = job

    dst_path = Path(dst_root) / f'{token}.{suffix}'
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists() and not overwrite:
        return 'skip', str(dst_path), 0

    lidar_path = Path(lidar_path)
    if not lidar_path.exists():
        raise FileNotFoundError(f'LiDAR file does not exist: {lidar_path}')

    points = _load_points(lidar_path, load_dim, use_dim)
    if filter_egocar_points:
        points = _filter_egocar_points(points, ego_width, ego_length)

    range_img = pts2range_np(
        points,
        lidar2ego=lidar2ego,
        **params,
    )
    range_imgs = range_img[None].astype(np.float32, copy=False)
    valid_pixels = int(np.count_nonzero(range_imgs))

    payload = dict(
        range_imgs=range_imgs,
        points_coord_after_transform='lidar',
        lidar_path=str(lidar_path),
        pc_range=np.asarray(
            params['pc_range']
            if params.get('pc_range') is not None else [], dtype=np.float32),
        pc_range_coord=params.get('pc_range_coord', 'ego'),
        range_image_shape=np.asarray(range_imgs.shape, dtype=np.int32),
    )

    np.savez_compressed(dst_path, **payload)
    return 'ok', str(dst_path), valid_pixels


def parse_args():
    parser = argparse.ArgumentParser(
        description='Precompute nuScenes raw LiDAR point clouds as range '
        'images for OccLiDARVAE.')
    parser.add_argument('ann_file', help='nuScenes info pkl.')
    parser.add_argument('dst_root', help='Output directory for token npz files.')
    parser.add_argument(
        '--data-root',
        default='data/nuscenes',
        help='Dataset root used when lidar_path in ann_file is relative.')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--suffix', default='npz')
    parser.add_argument('--H', type=int, default=32)
    parser.add_argument('--W', type=int, default=1024)
    parser.add_argument('--min-depth', type=float, default=1.45)
    parser.add_argument('--max-depth', type=float, default=80.0)
    parser.add_argument('--fov-up', type=float, default=10.0)
    parser.add_argument('--fov-down', type=float, default=-30.0)
    parser.add_argument(
        '--pc-range',
        type=float,
        nargs=6,
        default=DEFAULT_PC_RANGE,
        metavar=('X_MIN', 'Y_MIN', 'Z_MIN', 'X_MAX', 'Y_MAX', 'Z_MAX'),
        help='Point-cloud range used by OccLiDARVAE.')
    parser.add_argument(
        '--pc-range-coord',
        choices=['ego', 'lidar'],
        default='ego',
        help='Coordinate frame of --pc-range. Use ego for tokenizer configs.')
    parser.add_argument(
        '--no-pc-range-filter',
        action='store_true',
        help='Disable point filtering by --pc-range.')
    parser.add_argument('--load-dim', type=int, default=5)
    parser.add_argument(
        '--use-dim',
        type=int,
        nargs='+',
        default=[0, 1, 2, 3, 4],
        help='Point dimensions to keep from raw LiDAR file.')
    parser.add_argument(
        '--no-filter-egocar-points',
        action='store_true',
        help='Keep points on the ego vehicle.')
    parser.add_argument('--ego-width', type=float, default=1.85)
    parser.add_argument('--ego-length', type=float, default=4.084)
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing range npz files.')
    return parser.parse_args()


def main():
    args = parse_args()
    use_dim = list(args.use_dim)
    if len(use_dim) < 3:
        raise ValueError('--use-dim must include at least xyz dimensions.')
    if max(use_dim) >= args.load_dim:
        raise ValueError(
            f'All --use-dim values must be < --load-dim={args.load_dim}, '
            f'got {use_dim}.')

    params = dict(
        H=args.H,
        W=args.W,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        fov_up=args.fov_up,
        fov_down=args.fov_down,
        pc_range=None if args.no_pc_range_filter else args.pc_range,
        pc_range_coord=args.pc_range_coord,
    )

    infos = _load_infos(args.ann_file)
    if args.limit is not None:
        infos = infos[:args.limit]

    jobs = []
    for info in infos:
        token = info.get('token')
        if token is None:
            raise KeyError('Each info must contain a sample token.')
        lidar_path = _resolve_lidar_path(info, args.data_root)
        lidar2ego = _resolve_lidar2ego(info)
        jobs.append((
            token,
            str(lidar_path),
            args.dst_root,
            args.suffix,
            params,
            lidar2ego,
            args.load_dim,
            use_dim,
            not args.no_filter_egocar_points,
            args.ego_width,
            args.ego_length,
            args.overwrite,
        ))

    counts = {'ok': 0, 'skip': 0}
    valid_pixels_total = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for status, _, valid_pixels in executor.map(
                convert_one, jobs, chunksize=16):
            counts[status] = counts.get(status, 0) + 1
            valid_pixels_total += valid_pixels
            done = counts['ok'] + counts['skip']
            if done % 1000 == 0:
                print(
                    f'processed={done}/{len(jobs)} ok={counts["ok"]} '
                    f'skip={counts["skip"]}',
                    flush=True)

    avg_valid = valid_pixels_total / max(counts['ok'], 1)
    print(
        f'done total={len(jobs)} ok={counts["ok"]} skip={counts["skip"]} '
        f'avg_valid_pixels={avg_valid:.1f}')


if __name__ == '__main__':
    main()
