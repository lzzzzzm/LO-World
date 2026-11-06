# Copyright (c) OpenMMLab. All rights reserved.
import os
import copy
from os import path as osp
from typing import Callable, List, Union

import numpy as np

from mmdet3d.registry import DATASETS
from projects.datasets.base_dataset import BaseDataset

@DATASETS.register_module()
class NuScenesWorldDataset(BaseDataset):
    r"""NuScenes World Dataset.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict]): Pipeline used for data processing.
            Defaults to [].
        modality (dict): Modality to specify the sensor data used as input.
            Defaults to dict(use_camera=False, use_lidar=True).
        test_mode (bool): Whether the dataset is in test mode.
            Defaults to False.
    """
    METAINFO = {
        'classes':
        ('car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
         'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'),
        'version':
        'v1.0-trainval',
        'palette': [
            (255, 158, 0),  # Orange
            (255, 99, 71),  # Tomato
            (255, 140, 0),  # Darkorange
            (255, 127, 80),  # Coral
            (233, 150, 70),  # Darksalmon
            (220, 20, 60),  # Crimson
            (255, 61, 99),  # Red
            (0, 0, 230),  # Blue
            (47, 79, 79),  # Darkslategrey
            (112, 128, 144),  # Slategrey
        ]
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 modality: dict = dict(use_camera=False, use_lidar=True, use_occ=False),
                 data_prefix: dict = dict(),
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 load_interval: int = 1,
                 load_prev_number=None,
                 load_futu_number=None,
                 use_sequence_group_flag=False,
                 sequences_split_num=1,
                 **kwargs) -> None:
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            test_mode=test_mode,
            data_prefix=data_prefix,
            load_interval=load_interval,
            use_sequence_group_flag=use_sequence_group_flag,
            sequences_split_num=sequences_split_num,
            **kwargs)
        
        self.modality = modality
        self.load_prev_number = load_prev_number
        self.load_futu_number = load_futu_number

    def _pack_temporal_info(self, info: dict):
        result_info = dict(
            token=info['token'],
            ego2global=np.array(info['ego2global'])
        )
        if 'timestamp' in info:
            result_info['timestamp'] = info['timestamp']

        if 'lidar_points' in info:
            result_info['lidar2ego'] = np.array(
                info['lidar_points']['lidar2ego'])
            result_info['ego2lidar'] = np.linalg.inv(result_info['lidar2ego'])
            if self.modality.get('use_lidar', False):
                result_info['lidar_points'] = info['lidar_points']

        if self.modality.get('use_occ', False):
            result_info['occ_path'] = os.path.join(
                self.data_root, 'gts', info['scene_name'], info['token'],
                'labels.npz')
        return result_info

    def _get_info_from_index(self, target_idx, scene_name):
        info = self.get_data_info(target_idx)
        if info['scene_name'] != scene_name:
            return None

        return self._pack_temporal_info(info)

    def load_prev_info(self, index: int, current_scene_name: str):
        """
            Load the previous frames' information for the current sample, 
            which is used for the temporal modeling in the world model.
        """
        if self.load_prev_number is None or self.load_prev_number <= 0:
            return []
            
        prev_infos = []
        last_valid_info = None
        
        for i in range(1, self.load_prev_number + 1):
            target_idx = index - i
            info = None
            if target_idx >= 0:
                info = self._get_info_from_index(target_idx, current_scene_name)
            
            if info is not None:
                last_valid_info = info
                prev_infos.append(info)
            else:
                # Padding with the last valid frame if scene changes or out of bounds
                if last_valid_info is not None:
                    prev_infos.append(last_valid_info)
                else:
                    # If there is no previous valid frame at all, pad with current frame info
                    current_info = self._get_info_from_index(index, current_scene_name)
                    prev_infos.append(current_info)
                    
        return prev_infos[::-1] # return in chronological order

    def load_futu_info(self,
                       index: int,
                       current_scene_name: str,
                       return_padding_mask: bool = False):
        """
            Load the future frames' information for the current sample, 
            which is used for the temporal modeling in the world model.
        """
        if self.load_futu_number is None or self.load_futu_number <= 0:
            if return_padding_mask:
                return [], []
            return []
            
        futu_infos = []
        futu_padding_mask = []
        last_valid_info = None
        
        for i in range(1, self.load_futu_number + 1):
            target_idx = index + i
            info = None
            if target_idx < len(self):
                info = self._get_info_from_index(target_idx, current_scene_name)
            
            if info is not None:
                last_valid_info = info
                futu_infos.append(info)
                futu_padding_mask.append(False)
            else:
                # Padding with the last valid frame if scene changes or out of bounds
                if last_valid_info is not None:
                    futu_infos.append(last_valid_info)
                else:
                    # If there is no future valid frame at all, pad with current frame info
                    current_info = self._get_info_from_index(index, current_scene_name)
                    futu_infos.append(current_info)
                futu_padding_mask.append(True)
                    
        if return_padding_mask:
            return futu_infos, futu_padding_mask
        return futu_infos
    
    def prepare_data(self, index: int) -> Union[dict, None]:
        """Data preparation for both training and testing stage.

        Called by `__getitem__`  of dataset.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict or None: Data dict of the corresponding index.
        """
        # ----------- Get the raw input data dict from the annotation file -----------
        ori_input_dict = self.get_data_info(index)
        input_dict_raw = copy.deepcopy(ori_input_dict)

        # ---------- Reorganize the input_dict ------------------
        input_dict = dict(
            sample_idx=input_dict_raw['sample_idx'],
            token=input_dict_raw['token'],
            timestamp=input_dict_raw['timestamp'],
            scene_name=input_dict_raw['scene_name'],
            ego2global=np.array(input_dict_raw['ego2global']),  # [4, 4]
        )
        for key in ('sample_index', 'sequence_group_idx',
                    'sequence_frame_idx', 'sequence_group_length',
                    'start_of_sequence'):
            if key in input_dict_raw:
                input_dict[key] = input_dict_raw[key]
        if 'instances' in input_dict_raw:
            input_dict['instances'] = input_dict_raw['instances']

        # Images-related information
        if self.modality['use_camera']:
            cam2imgs, cam2egos, lidar2cams = [], [], []
            for cam_type in input_dict_raw['images'].keys():
                cam_info = input_dict_raw['images'][cam_type]
                cam2imgs.append(cam_info['cam2img'])
                cam2egos.append(cam_info['cam2ego'])
                lidar2cams.append(cam_info['lidar2cam'])
            input_dict['cam2imgs'] = np.stack(cam2imgs, axis=0)     # [N, 3, 3]
            input_dict['cam2egos'] = np.stack(cam2egos, axis=0)     # [N, 4, 4]
            input_dict['lidar2cams'] = np.stack(lidar2cams, axis=0) # [N, 4, 4]
        
        # LiDAR-related information
        if 'lidar_points' in input_dict_raw:
            input_dict['lidar2ego'] = np.array(input_dict_raw['lidar_points']['lidar2ego']) # [4, 4]
            input_dict['ego2lidar'] = np.linalg.inv(input_dict['lidar2ego']) # [4, 4]
            if self.modality['use_lidar']:
                input_dict['lidar_points'] = input_dict_raw['lidar_points']
        
        # Occ-related information
        if self.modality.get('use_occ', False):
            # Occ3d
            input_dict['occ_path'] = os.path.join(
                self.data_root, 'gts', input_dict_raw['scene_name'], input_dict_raw['token'], 'labels.npz'
            )
        
        # ----------------------- Load Temporal Information if needed -----------------------
        # Info order is t-P, ..., t-1, t, t+1, ..., t+F

        curr_ego2global = input_dict['ego2global']

        # Load previous frames' information
        if self.load_prev_number is not None and self.load_prev_number > 0:
            prev_info = self.load_prev_info(index, input_dict['scene_name'])
            curr2prev_egos = []
            curr2prev_egos_relative = []

            all_prev_egos = [p['ego2global'] for p in prev_info] + [curr_ego2global]

            for i, p_info in enumerate(prev_info):
                curr2prev_ego = np.linalg.inv(p_info['ego2global']) @ curr_ego2global
                curr2prev_egos.append(curr2prev_ego)

                rel_ego = np.linalg.inv(all_prev_egos[i]) @ all_prev_egos[i+1]
                curr2prev_egos_relative.append(rel_ego)

            input_dict['curr2prev_ego'] = np.stack(curr2prev_egos, axis=0)                   # [P, 4, 4], t->t-P, t->t-(P-1), ..., t->t-1
            input_dict['curr2prev_ego_relative'] = np.stack(curr2prev_egos_relative, axis=0) # [P, 4, 4], t-(P-1)->t-P, ..., t-1->t-2, t->t-1
            input_dict['prev_info'] = prev_info
        
        # Load future frames' information
        if self.load_futu_number is not None and self.load_futu_number > 0:
            if 'futu_info' in input_dict_raw:
                futu_info = []
                futu_padding_mask = []
                last_valid_info = None
                current_info = self._pack_temporal_info(input_dict_raw)
                seen_tokens = {input_dict['token']}
                raw_futu_infos = input_dict_raw['futu_info'][
                    :self.load_futu_number]
                for step in range(self.load_futu_number):
                    raw_info = (
                        raw_futu_infos[step]
                        if step < len(raw_futu_infos) else None)
                    raw_scene_name = (
                        raw_info.get('scene_name')
                        if raw_info is not None else None)
                    is_valid = (
                        raw_info is not None and
                        (raw_scene_name is None or
                         raw_scene_name == input_dict['scene_name']))
                    if is_valid:
                        packed_info = self._pack_temporal_info(raw_info)
                        is_padding = packed_info['token'] in seen_tokens
                        seen_tokens.add(packed_info['token'])
                        futu_info.append(packed_info)
                        futu_padding_mask.append(is_padding)
                        if not is_padding:
                            last_valid_info = packed_info
                    else:
                        if last_valid_info is not None:
                            futu_info.append(last_valid_info)
                        else:
                            futu_info.append(current_info)
                        futu_padding_mask.append(True)
            else:
                futu_info, futu_padding_mask = self.load_futu_info(
                    index,
                    input_dict['scene_name'],
                    return_padding_mask=True)
            curr2futu_egos = []
            rel_futu_egos = []

            all_futu_egos = [curr_ego2global] + [f['ego2global'] for f in futu_info]

            for i, f_info in enumerate(futu_info):
                curr2futu_ego = np.linalg.inv(f_info['ego2global']) @ curr_ego2global
                curr2futu_egos.append(curr2futu_ego)

                rel_ego = np.linalg.inv(all_futu_egos[i]) @ all_futu_egos[i+1]
                rel_futu_egos.append(rel_ego)

            input_dict['curr2futu_ego'] = np.stack(curr2futu_egos, axis=0)         # [F, 4, 4], t->t+1, t->t+2, ..., t->t+F
            input_dict['curr2futu_ego_relative'] = np.stack(rel_futu_egos, axis=0) # [F, 4, 4], t->t+1, t+1->t+2, ..., t+(F-1)->t+F
            input_dict['futu_info'] = futu_info
            input_dict['futu_padding_mask'] = np.asarray(
                futu_padding_mask, dtype=np.bool_)
        
        total_token = [p['token'] for p in input_dict.get('prev_info', [])] + \
                      [input_dict['token']] + \
                      [f['token'] for f in input_dict.get('futu_info', [])]
        input_dict['total_token'] = total_token
        total_infos = (
            input_dict.get('prev_info', []) + [input_dict] +
            input_dict.get('futu_info', []))
        input_dict['total_ego2global'] = np.stack(
            [info['ego2global'] for info in total_infos], axis=0)
        if all('timestamp' in info for info in total_infos):
            input_dict['total_timestamp'] = np.array(
                [info['timestamp'] for info in total_infos],
                dtype=np.float64)
        if all('lidar2ego' in info for info in total_infos):
            input_dict['total_lidar2ego'] = np.stack(
                [info['lidar2ego'] for info in total_infos], axis=0)

        # -------------- apply data pipeline ----------------
        example = self.pipeline(input_dict)    

        return example
