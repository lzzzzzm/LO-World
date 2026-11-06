# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch

from numbers import Number
from typing import Dict, List, Optional, Sequence, Tuple, Union

from mmdet.models.utils.misc import samplelist_boxtype2tensor

from mmdet3d.registry import MODELS
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmdet3d.models.data_preprocessors import Det3DDataPreprocessor
from mmdet3d.utils import OptConfigType

@MODELS.register_module()
@MMENGINE_MODELS.register_module()
class MMDataPreprocessor(Det3DDataPreprocessor):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    
    def simple_process(self, data: dict, training: bool = False) -> dict:
        """
        For Img:
            Perform normalization, padding and bgr2rgb conversion for img data
            based on ``BaseDataPreprocessor``
        For Point Cloud:
            voxelize point cloud if `voxel` is set to be True.
        For Occ:
            Peform concatenation of voxel semantics
        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.
                Defaults to False.

        Returns:
            dict: Data in the same format as the model input.
        """
        if 'img' in data['inputs']:
            batch_pad_shape = self._get_pad_shape(data)

        data = self.collate_data(data)
        inputs, data_samples = data['inputs'], data['data_samples']
        batch_inputs = dict()

        if 'points' in inputs:
            batch_inputs['points'] = inputs['points']

            if self.voxel:
                voxel_dict = self.voxelize(inputs['points'], data_samples)
                batch_inputs['voxels'] = voxel_dict
        
        for key, dtype in [('voxel_semantics', torch.long), ('bev_latent', torch.float), ('bev_layout', torch.float)]:
            if key in inputs:
                if isinstance(inputs[key], list):
                    batch_inputs[key] = torch.stack(inputs[key], dim=0).to(dtype)
                else:
                    raise ValueError(f"Unsupported type for {key}: {type(inputs[key])}")

        for key, dtype in [('pred_range_imgs', torch.float),
                           ('range_imgs', torch.float),
                           ('box_range_layout', torch.float),
                           ('box_tokens', torch.float)]:
            if key in inputs:
                if isinstance(inputs[key], list):
                    batch_inputs[key] = torch.stack(inputs[key], dim=0).to(dtype)
                else:
                    raise ValueError(f"Unsupported type for {key}: {type(inputs[key])}")

        if 'imgs' in inputs:
            imgs = inputs['imgs']

            if data_samples is not None:
                # NOTE the batched image size information may be useful, e.g.
                # in DETR, this is needed for the construction of masks, which
                # is then used for the transformer_head.
                batch_input_shape = tuple(imgs[0].size()[-2:])
                for data_sample, pad_shape in zip(data_samples,
                                                  batch_pad_shape):
                    data_sample.set_metainfo({
                        'batch_input_shape': batch_input_shape,
                        'pad_shape': pad_shape
                    })

                if self.boxtype2tensor:
                    samplelist_boxtype2tensor(data_samples)
                if self.pad_mask:
                    self.pad_gt_masks(data_samples)
                if self.pad_seg:
                    self.pad_gt_sem_seg(data_samples)

            if training and self.batch_augments is not None:
                for batch_aug in self.batch_augments:
                    imgs, data_samples = batch_aug(imgs, data_samples)
            batch_inputs['imgs'] = imgs

        return {'inputs': batch_inputs, 'data_samples': data_samples}
