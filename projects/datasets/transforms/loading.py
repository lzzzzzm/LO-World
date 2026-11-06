import os
import numpy as np
import copy
from collections.abc import Mapping
from typing import List, Optional, Union

import mmcv
import mmengine
from mmengine.fileio import get

from mmcv import BaseTransform
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.bbox_3d import get_box_type
from mmdet3d.structures.points import BasePoints, get_points_type

from .utils import pts2range_np


@TRANSFORMS.register_module()
class LoadStreamOcc3DFromFile(BaseTransform):
    """
        Load stream occupancy data from files.
        utilize the LoadOcc3DFromFile transform to load occupancy data for each timestamp in the input stream, and then concatenate the voxel semantics along the time dimension to form a 4D tensor of shape (B, F, Y, X, Z), where B is the batch size, F is the number of timestamps in the input stream, and Y, X, Z are the BEV-y, BEV-x and vertical voxel axes.

        Args:
            camera_mask (bool): If true, set labels outside each frame's
                ``mask_camera`` to ``ignore_index``.
            ignore_index (int): Label used for voxels excluded by the mask.
    """
    def __init__(self, to_long=True, camera_mask=False, ignore_index=255):
        self.to_long = to_long
        self.camera_mask = camera_mask
        self.ignore_index = ignore_index
        self.load_occ3d_from_file = LoadOcc3DFromFile(to_long=to_long)
    
    def transform(self, results):
        prev_infos = results.get('prev_info', [])
        futu_infos = results.get('futu_info', [])
        all_infos = prev_infos + [results] + futu_infos

        voxel_semantics_list = []
        for info in all_infos:
            voxel_semantics = self.load_occ3d_from_file.transform(info)['voxel_semantics']
            if self.camera_mask:
                with np.load(info['occ_path']) as occ3d_label:
                    if 'mask_camera' not in occ3d_label:
                        raise KeyError(
                            f"{info['occ_path']} does not contain 'mask_camera'")
                    camera_mask = occ3d_label['mask_camera'].astype(bool)
                if camera_mask.shape != voxel_semantics.shape:
                    raise ValueError(
                        f"mask_camera shape {camera_mask.shape} does not match "
                        f"semantics shape {voxel_semantics.shape} in "
                        f"{info['occ_path']}")
                voxel_semantics = voxel_semantics.copy()
                voxel_semantics[~camera_mask] = self.ignore_index
            voxel_semantics_list.append(voxel_semantics)
        
        results['voxel_semantics'] = np.stack(voxel_semantics_list, axis=0) # (F, Y, X, Z)

        return results

@TRANSFORMS.register_module()
class LoadStreamDataFromFile(BaseTransform):
    """
        Load stream data (e.g., BEV layout, BEV latent features) from files.
        utilize the LoadDataFromFile transform to load different types of data based on the keys present in the results dictionary.
    """
    def __init__(
            self, 
            data_root, 
            input_keys=None,
            output_keys=None,
            suffix=None,
            current_only=False,
        ):
        self.data_root = data_root
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.suffix = suffix
        self.current_only = current_only
        self.load_data_from_file = LoadDataFromFile(
            data_root=data_root,
            input_keys=input_keys,
            output_keys=output_keys,
            suffix=suffix,
        )
    
    def transform(self, results):
        if self.current_only:
            all_infos = [results]
        else:
            prev_infos = results.get('prev_info', [])
            futu_infos = results.get('futu_info', [])
            all_infos = prev_infos + [results] + futu_infos

        data_list = []
        for info in all_infos:
            data = self.load_data_from_file.transform(info)[self.output_keys]
            data_list.append(data)
        results[self.output_keys] = np.stack(data_list, axis=0)
        
        return results

@TRANSFORMS.register_module()
class LoadDataFromFile(BaseTransform):
    """
        A general data loading transform that can be used to load various types of data (e.g., BEV layout, BEV latent features) from files. 
        The specific type of data to load is determined by the keys present in the results dictionary. 
        This transform checks for the presence of specific keys (e.g., 'bev_layout_path', 'bev_latent_path') and loads the corresponding data accordingly.
    """
    def __init__(
            self, 
            data_root, 
            input_keys=None,
            output_keys=None,
            suffix=None,
        ):
        self.data_root = data_root
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.suffix = suffix
    
    def transform(self, results):
        token = results['token']
        data_path = os.path.join(self.data_root, token + f'.{self.suffix}')
        if self.suffix == 'npz':
            with np.load(data_path) as data:
                results[self.output_keys] = data[self.input_keys]
        elif self.suffix == 'npy':
            results[self.output_keys] = np.load(data_path)

        return results

@TRANSFORMS.register_module()
class LoadOcc3DFromFile(BaseTransform):
    """Load occupancy data from the dataset path or an alternate root.

    Args:
        to_long (bool): Convert labels to ``int64``.
        data_root (str, optional): If set, load
            ``<data_root>/<scene_name>/<token>/labels.npz`` instead of the
            dataset-provided ``occ_path``. This is useful for evaluating an
            externally predicted occupancy source without changing targets.
        output_key (str): Results key where the semantic grid is stored.
    """

    def __init__(self, to_long=True, data_root=None,
                 output_key='voxel_semantics'):
        self.to_long = to_long
        self.data_root = data_root
        self.output_key = output_key

    def transform(self, results):
        if self.data_root is None:
            occ3d_path = results['occ_path']
        else:
            occ3d_path = os.path.join(
                self.data_root, results['scene_name'], results['token'],
                'labels.npz')
        with np.load(occ3d_path) as occ3d_label:
            if 'semantics' in occ3d_label:
                voxel_semantics = occ3d_label['semantics']
            elif 'voxel_label' in occ3d_label:
                # Occ3D-Waymo uses ``voxel_label`` for the semantic grid.
                voxel_semantics = occ3d_label['voxel_label']
            else:
                raise KeyError(
                    f'{occ3d_path} contains neither semantics nor '
                    'voxel_label')

        if self.to_long:
            voxel_semantics = voxel_semantics.astype(np.int64)
        results[self.output_key] = voxel_semantics

        return results


@TRANSFORMS.register_module()
class RemapOccLabels(BaseTransform):
    """Remap occupancy labels with an explicit lookup table.

    This is primarily intended for cross-dataset evaluation, where the model
    output taxonomy must remain unchanged. Labels outside the lookup table are
    assigned ``unknown_label`` instead of being silently clipped.
    """

    def __init__(self, mapping, unknown_label=255):
        self.unknown_label = int(unknown_label)
        if isinstance(mapping, Mapping):
            if any(int(key) < 0 for key in mapping):
                raise ValueError('mapping keys must be non-negative labels')
            size = max((int(key) for key in mapping), default=-1) + 1
            lookup = np.full(size, self.unknown_label, dtype=np.int64)
            for source, target in mapping.items():
                lookup[int(source)] = int(target)
            mapping = lookup
        self.mapping = np.asarray(mapping, dtype=np.int64)
        if self.mapping.ndim != 1:
            raise ValueError(
                'mapping must be a dict or one-dimensional sequence')

    def transform(self, results):
        labels = np.asarray(results['voxel_semantics'])
        remapped = np.full(labels.shape, self.unknown_label, dtype=np.int64)
        valid = (labels >= 0) & (labels < self.mapping.size)
        remapped[valid] = self.mapping[labels[valid]]
        results['voxel_semantics'] = remapped
        return results


@TRANSFORMS.register_module()
class PointsToRangeImage(BaseTransform):
    """Project loaded points to the tokenizer's depth range image.

    ``pc_range`` is applied in the coordinate system selected by
    ``pc_range_coord``. For the Waymo extraction in this repository both the
    stored cloud and occupancy volume use the vehicle/ego frame.
    """

    def __init__(self,
                 height=32,
                 width=1024,
                 min_depth=1.45,
                 max_depth=80.0,
                 fov_up=10.0,
                 fov_down=-30.0,
                 points_coord='lidar',
                 pc_range=None,
                 pc_range_coord='ego',
                 filter_out_of_fov=False,
                 output_key='range_imgs'):
        if points_coord not in ('ego', 'lidar'):
            raise ValueError('points_coord must be "ego" or "lidar"')
        self.height = int(height)
        self.width = int(width)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.fov_up = float(fov_up)
        self.fov_down = float(fov_down)
        self.points_coord = points_coord
        self.pc_range = pc_range
        self.pc_range_coord = pc_range_coord
        self.filter_out_of_fov = bool(filter_out_of_fov)
        self.output_key = output_key

    def transform(self, results):
        points = results['points']
        if isinstance(points, BasePoints):
            points = points.tensor.detach().cpu().numpy()
        else:
            points = np.asarray(points)

        ego2lidar = None
        lidar2ego = None
        if self.points_coord == 'ego':
            ego2lidar = np.asarray(results['ego2lidar'], dtype=np.float32)
        else:
            lidar2ego = np.asarray(results['lidar2ego'], dtype=np.float32)

        results[self.output_key] = pts2range_np(
            points,
            ego2lidar=ego2lidar,
            lidar2ego=lidar2ego,
            pc_range=self.pc_range,
            pc_range_coord=self.pc_range_coord,
            H=self.height,
            W=self.width,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            fov_up=self.fov_up,
            fov_down=self.fov_down,
            filter_out_of_fov=self.filter_out_of_fov)
        return results


@TRANSFORMS.register_module()
class LoadStreamRangeImageFromPoints(BaseTransform):
    """Load and project every temporal point cloud to a range image."""

    def __init__(self,
                 load_dim=5,
                 use_dim=5,
                 filter_egocar_points=True,
                 filter_egocar_coord='stored',
                 pc_range=None,
                 **range_image_kwargs):
        self.point_loader = CustomLoadPointsFromFile(
            coord_type='LIDAR',
            load_dim=load_dim,
            use_dim=use_dim,
            filter_egocar_points=filter_egocar_points,
            filter_egocar_coord=filter_egocar_coord,
            # Range filtering belongs to PointsToRangeImage, which knows
            # whether the requested volume is expressed in ego or LiDAR
            # coordinates. Filtering here would blindly use the stored point
            # coordinates and can discard valid points before conversion.
            pc_range=None)
        self.range_projector = PointsToRangeImage(
            pc_range=pc_range, **range_image_kwargs)

    def transform(self, results):
        prev_infos = results.get('prev_info', [])
        futu_infos = results.get('futu_info', [])
        all_infos = prev_infos + [results] + futu_infos
        range_images = []
        for info in all_infos:
            frame = self.point_loader.transform(copy.copy(info))
            frame = self.range_projector.transform(frame)
            range_images.append(frame[self.range_projector.output_key])
        results[self.range_projector.output_key] = np.stack(
            range_images, axis=0)
        return results

@TRANSFORMS.register_module()
class CustomLoadPointsFromFile(BaseTransform):
    """Load Points From File.

    Required Keys:

    - lidar_points (dict)

        - lidar_path (str)

    Added Keys:

    - points (np.float32)

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:

            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points. Defaults to 6.
        use_dim (list[int] | int): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
        norm_intensity (bool): Whether to normlize the intensity. Defaults to
            False.
        norm_elongation (bool): Whether to normlize the elongation. This is
            usually used in Waymo dataset.Defaults to False.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(self,
                 coord_type: str,
                 load_dim: int = 6,
                 use_dim: Union[int, List[int]] = [0, 1, 2],
                 shift_height: bool = False,
                 use_color: bool = False,
                 norm_intensity: bool = False,
                 norm_elongation: bool = False,
                 to_ego: bool = False,
                 filter_egocar_points: bool = True,
                 filter_egocar_coord: str = 'stored',
                 pc_range: Optional[List[float]] = None,
                 backend_args: Optional[dict] = None) -> None:
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.norm_intensity = norm_intensity
        self.norm_elongation = norm_elongation
        self.backend_args = backend_args
        self.to_ego = to_ego
        self.pc_range = pc_range
        self.filter_egocar_points = filter_egocar_points
        if filter_egocar_coord not in ('stored', 'ego'):
            raise ValueError(
                'filter_egocar_coord must be "stored" or "ego"')
        self.filter_egocar_coord = filter_egocar_coord
        self.ego_width, self.ego_length = 1.85, 4.084

    def _load_points(self, pts_filename: str) -> np.ndarray:
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        try:
            pts_bytes = get(pts_filename, backend_args=self.backend_args)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmengine.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    @staticmethod
    def _ring_ids_from_feature(points: np.ndarray, ring_index: int,
                               source_beams: int):
        """Return normalized zero-based ring ids, or ``None`` if invalid."""
        if not -points.shape[1] <= ring_index < points.shape[1]:
            return None
        values = points[:, ring_index]
        finite = np.isfinite(values)
        rounded = np.rint(values)
        integer = finite & (np.abs(values - rounded) < 1e-3)
        if not integer.any():
            return None
        ring_ids = rounded.astype(np.int64)
        valid_values = ring_ids[integer]
        if valid_values.min() >= 1 and valid_values.max() <= source_beams:
            ring_ids = ring_ids - 1
            valid_values = ring_ids[integer]
        valid = integer & (ring_ids >= 0) & (ring_ids < source_beams)
        if (valid.sum() != points.shape[0] or
                np.unique(valid_values).size < 2):
            return None
        return ring_ids

    @staticmethod
    def _ring_ids_from_elevation(points: np.ndarray, config: dict):
        xyz = points[:, :3]
        ego2lidar = config.get('ego2lidar')
        if ego2lidar is not None:
            ego2lidar = np.asarray(ego2lidar, dtype=np.float32)
            xyz_hom = np.concatenate(
                [xyz, np.ones((xyz.shape[0], 1), dtype=xyz.dtype)], axis=1)
            xyz = (ego2lidar @ xyz_hom.T).T[:, :3]
        else:
            origin = np.asarray(config.get('origin', (0.0, 0.0, 0.0)),
                                dtype=np.float32)
            xyz = xyz - origin
        horizontal_range = np.linalg.norm(xyz[:, :2], axis=1)
        pitch = np.degrees(np.arctan2(xyz[:, 2], horizontal_range))
        fov_up = float(config['fov_up'])
        fov_down = float(config['fov_down'])
        source_beams = int(config['source_beams'])
        row = np.floor(
            (fov_up - pitch) / (fov_up - fov_down) * source_beams)
        valid = np.isfinite(row) & (row >= 0) & (row < source_beams)
        return row.astype(np.int64), valid

    def _sample_lidar_beams(self, points: np.ndarray, config: dict):
        source_beams = int(config['source_beams'])
        target_beams = int(config['target_beams'])
        if source_beams % target_beams != 0:
            raise ValueError(
                'source_beams must be divisible by target_beams')
        stride = source_beams // target_beams
        offset = int(config.get('offset', 0))
        method = config.get('method', 'auto')
        ring_ids = None

        ring_index = config.get('ring_index')
        if method in ('auto', 'ring') and ring_index is not None:
            ring_ids = self._ring_ids_from_feature(
                points, int(ring_index), source_beams)
        if method == 'ring' and ring_ids is None:
            raise ValueError(
                'Configured ring feature does not contain valid ring ids')

        if ring_ids is None:
            ring_ids, valid = self._ring_ids_from_elevation(points, config)
            used_method = 'elevation'
        else:
            valid = np.ones(points.shape[0], dtype=np.bool_)
            used_method = 'ring'
        keep = valid & ((ring_ids % stride) == offset)
        return points[keep], used_method

    def transform(self, results: dict) -> dict:
        """Method to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
            Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_file_path = results['lidar_points']['lidar_path']
        points = self._load_points(pts_file_path)
        points = points.reshape(-1, self.load_dim)
        beam_sampling = results['lidar_points'].get('beam_sampling')
        if beam_sampling is not None:
            input_point_count = points.shape[0]
            points, sampling_method = self._sample_lidar_beams(
                points, beam_sampling)
            results['lidar_beam_sampling'] = {
                'method': sampling_method,
                'source_beams': int(beam_sampling['source_beams']),
                'target_beams': int(beam_sampling['target_beams']),
                'input_points': input_point_count,
                'output_points': points.shape[0],
            }
        points = points[:, self.use_dim]

        if self.filter_egocar_points:
            filter_xyz = points[:, :3]
            if self.filter_egocar_coord == 'ego':
                if 'lidar2ego' not in results:
                    raise KeyError(
                        'lidar2ego is required for ego-coordinate self-car '
                        'filtering')
                lidar2ego = np.asarray(results['lidar2ego'], dtype=np.float32)
                xyz_hom = np.concatenate([
                    filter_xyz,
                    np.ones((filter_xyz.shape[0], 1), dtype=filter_xyz.dtype),
                ], axis=1)
                filter_xyz = (lidar2ego @ xyz_hom.T).T[:, :3]
            mask = np.logical_or(
                np.abs(filter_xyz[:, 1]) > self.ego_width / 2,
                np.logical_or(
                    filter_xyz[:, 0] > self.ego_length / 2 + 1.0,
                    filter_xyz[:, 0] < -self.ego_length / 2 + 0.5)
            )
            points = points[mask]

        if self.to_ego:
            assert 'lidar2ego' in results, 'To transform points to ego coordinates, '
            lidar2ego = results['lidar2ego']    # 4x4
            points_last = points[:, 3:]
            points = (lidar2ego[:3, :3] @ points[:, :3].T).T + lidar2ego[:3, 3]
            points = np.concatenate((points, points_last), axis=1)

        if self.pc_range is not None:
            x_min, y_min, z_min, x_max, y_max, z_max = self.pc_range
            mask = (points[:, 0] > x_min) & (points[:, 0] < x_max) & \
                   (points[:, 1] > y_min) & (points[:, 1] < y_max) & \
                   (points[:, 2] > z_min) & (points[:, 2] < z_max)
            points = points[mask]

        if self.norm_intensity:
            assert len(self.use_dim) >= 4, \
                f'When using intensity norm, expect used dimensions >= 4, got {len(self.use_dim)}'  # noqa: E501
            points[:, 3] = np.tanh(points[:, 3])
        if self.norm_elongation:
            assert len(self.use_dim) >= 5, \
                f'When using elongation norm, expect used dimensions >= 5, got {len(self.use_dim)}'  # noqa: E501
            points[:, 4] = np.tanh(points[:, 4])
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points

        return results
