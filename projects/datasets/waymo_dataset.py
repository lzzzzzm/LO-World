# Copyright (c) OpenMMLab. All rights reserved.
"""World-model dataset for the final Waymo TOP64x2048 protocol."""

import copy
import json
from os import path as osp
from typing import List, Union

import mmengine
import numpy as np

from mmdet3d.registry import DATASETS
from .nuscenes_dataset import NuScenesWorldDataset


@DATASETS.register_module()
class WaymoWorldDataset(NuScenesWorldDataset):
    """Load fixed 2Hz infos and TOP-frame five-feature point clouds.

    ``lidar_root`` contains <sample_id:07d>.bin files. Each sample's real
    lidar2ego comes from ``lidar_extrinsics_file``. Paths are relative to
    data_root; OCC labels use <occ_root>/<scene:03d>/<frame:03d>_04.npz.
    """

    METAINFO = {
        'classes': ('Car', 'Pedestrian', 'Cyclist', 'Sign'),
        'version': 'waymo-validation',
    }

    def __init__(self,
                 lidar_root: str,
                 lidar_extrinsics_file: str,
                 occ_root: str = 'validation',
                 **kwargs) -> None:
        self.occ_root = occ_root
        self.lidar_root = lidar_root
        # BaseDataset loads annotations inside super().__init__.
        manifest_path = osp.join(kwargs['data_root'], lidar_extrinsics_file)
        with open(manifest_path, encoding='utf-8') as stream:
            manifest = json.load(stream)
        self.sample_lidar2ego = {}
        for record in manifest['results']:
            transform = np.asarray(record['top_lidar2ego'], dtype=np.float32)
            for sample_id in record['sample_ids']:
                self.sample_lidar2ego[int(sample_id)] = transform
        super().__init__(**kwargs)

    def load_data_list(self) -> List[dict]:
        """Convert the fixed legacy infos to the world-model data format."""
        annotations = mmengine.load(self.ann_file)
        return [self._convert_legacy_info(info)
                for info in annotations[::self.load_interval]]

    def _convert_legacy_info(self, raw_info: dict) -> dict:
        sample_id = int(raw_info['image']['image_idx'])
        scene_idx, frame_idx = (sample_id // 1000) % 1000, sample_id % 1000
        scene_name = f'{scene_idx:03d}'
        return {
            'sample_id': sample_id,
            'token': f'{scene_name}_{frame_idx:03d}',
            'timestamp': raw_info['timestamp'],
            'scene_name': scene_name,
            'frame_idx': frame_idx,
            'ego2global': np.asarray(raw_info['pose'], dtype=np.float64),
            'occ_path': osp.join(
                self.data_root, self.occ_root, scene_name,
                f'{frame_idx:03d}_04.npz'),
            'lidar_points': {
                'lidar_path': osp.normpath(osp.join(
                    self.data_root, self.lidar_root, f'{sample_id:07d}.bin')),
                'lidar2ego': self.sample_lidar2ego[sample_id].copy(),
                'num_pts_feats': 5,
            },
        }

    def _pack_temporal_info(self, info: dict) -> dict:
        """Pack temporal fields while retaining Waymo's explicit paths."""
        result = super()._pack_temporal_info(info)
        if self.modality.get('use_occ', False):
            result['occ_path'] = info['occ_path']
        return result

    def prepare_data(self, index: int) -> Union[dict, None]:
        """Prepare a normalized Waymo sample and apply the data pipeline."""
        input_dict_raw = copy.deepcopy(self.get_data_info(index))
        input_dict = {
            'sample_idx': input_dict_raw['sample_idx'],
            'token': input_dict_raw['token'],
            'timestamp': input_dict_raw['timestamp'],
            'scene_name': input_dict_raw['scene_name'],
            'ego2global': np.asarray(input_dict_raw['ego2global']),
        }
        for key in ('sample_index', 'sequence_group_idx',
                    'sequence_frame_idx', 'sequence_group_length',
                    'start_of_sequence'):
            if key in input_dict_raw:
                input_dict[key] = input_dict_raw[key]

        lidar_points = input_dict_raw['lidar_points']
        input_dict['lidar2ego'] = np.asarray(lidar_points['lidar2ego'])
        input_dict['ego2lidar'] = np.linalg.inv(input_dict['lidar2ego'])
        if self.modality['use_lidar']:
            input_dict['lidar_points'] = lidar_points

        if self.modality.get('use_occ', False):
            input_dict['occ_path'] = input_dict_raw['occ_path']

        curr_ego2global = input_dict['ego2global']
        if self.load_prev_number is not None and self.load_prev_number > 0:
            prev_info = self.load_prev_info(index, input_dict['scene_name'])
            all_egos = ([p['ego2global'] for p in prev_info] +
                        [curr_ego2global])
            input_dict['curr2prev_ego'] = np.stack([
                np.linalg.inv(info['ego2global']) @ curr_ego2global
                for info in prev_info
            ])
            input_dict['curr2prev_ego_relative'] = np.stack([
                np.linalg.inv(all_egos[i]) @ all_egos[i + 1]
                for i in range(len(prev_info))
            ])
            input_dict['prev_info'] = prev_info

        if self.load_futu_number is not None and self.load_futu_number > 0:
            futu_info, padding_mask = self.load_futu_info(
                index, input_dict['scene_name'], return_padding_mask=True)
            all_egos = ([curr_ego2global] +
                        [f['ego2global'] for f in futu_info])
            input_dict['curr2futu_ego'] = np.stack([
                np.linalg.inv(info['ego2global']) @ curr_ego2global
                for info in futu_info
            ])
            input_dict['curr2futu_ego_relative'] = np.stack([
                np.linalg.inv(all_egos[i]) @ all_egos[i + 1]
                for i in range(len(futu_info))
            ])
            input_dict['futu_info'] = futu_info
            input_dict['futu_padding_mask'] = np.asarray(
                padding_mask, dtype=np.bool_)

        all_infos = (input_dict.get('prev_info', []) + [input_dict] +
                     input_dict.get('futu_info', []))
        input_dict['total_token'] = [info['token'] for info in all_infos]
        input_dict['total_ego2global'] = np.stack(
            [info['ego2global'] for info in all_infos])
        input_dict['total_timestamp'] = np.asarray(
            [info['timestamp'] for info in all_infos], dtype=np.float64)
        input_dict['total_lidar2ego'] = np.stack(
            [info['lidar2ego'] for info in all_infos])

        return self.pipeline(input_dict)
