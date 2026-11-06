# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Sequence, Union

import mmengine
import numpy as np
import torch
from mmcv import BaseTransform
from numpy import dtype

from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures import Det3DDataSample


def to_tensor(
    data: Union[torch.Tensor, np.ndarray, Sequence, int,
                float]) -> torch.Tensor:
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.

    Returns:
        torch.Tensor: the converted data.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        if data.dtype is dtype('float64'):
            data = data.astype(np.float32)
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmengine.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


@TRANSFORMS.register_module()
class MMPack3DDetInputs(BaseTransform):
    INPUTS_KEYS = (
        'voxel_semantics', 'bev_latent', 'bev_layout', 'range_imgs')

    def __init__(
        self,
        keys: tuple,
        meta_keys: tuple = (),
    ) -> None:
        unsupported_keys = set(keys) - set(self.INPUTS_KEYS)
        if unsupported_keys:
            raise ValueError(
                f'unsupported input keys: {sorted(unsupported_keys)}; '
                f'supported keys are {self.INPUTS_KEYS}')
        self.keys = tuple(keys)
        self.meta_keys = tuple(meta_keys)

    def transform(self, results: Union[dict,
                                       List[dict]]) -> Union[dict, List[dict]]:
        """Method to pack the input data. when the value in this dict is a
        list, it usually is in Augmentations Testing.

        Args:
            results (dict | list[dict]): Result dict from the data pipeline.

        Returns:
            dict | List[dict]:

            - 'inputs' (dict): Tensor inputs requested by ``self.keys``.

            - 'data_samples' (:obj:`Det3DDataSample`): The annotation info of
              the sample.
        """
        # augtest
        if isinstance(results, list):
            if len(results) == 1:
                # simple test
                return self.pack_single_results(results[0])
            pack_results = []
            for single_result in results:
                pack_results.append(self.pack_single_results(single_result))
            return pack_results
        # norm training and simple testing
        elif isinstance(results, dict):
            return self.pack_single_results(results)
        else:
            raise NotImplementedError

    def pack_single_results(self, results: dict) -> dict:
        """Method to pack the single input data. when the value in this dict is
        a list, it usually is in Augmentations Testing.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict: A dict contains

            - 'inputs' (dict): Tensor inputs requested by ``self.keys``.

            - 'data_samples' (:obj:`Det3DDataSample`): The annotation info
              of the sample.
        """
        for key in self.INPUTS_KEYS:
            if key in results:
                results[key] = to_tensor(results[key])

        data_sample = Det3DDataSample()

        data_metas = {
            key: results[key] for key in self.meta_keys if key in results
        }
        data_sample.set_metainfo(data_metas)

        inputs = {key: results[key] for key in self.keys if key in results}

        packed_results = dict()
        packed_results['data_samples'] = data_sample
        packed_results['inputs'] = inputs

        return packed_results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(keys={self.keys})'
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str
