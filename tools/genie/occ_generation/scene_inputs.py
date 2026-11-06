"""Scene-only nuScenes annotations and cached inference inputs.

The inherited dataset still constructs temporal metadata and padding. Full
physical scenes are retained so their starts and warmup semantics are unchanged.
"""

from __future__ import annotations

from collections import OrderedDict
import copy
import logging
from pathlib import Path
import pickle

import numpy as np
import torch

from projects.datasets.nuscenes_dataset import NuScenesWorldDataset
from projects.datasets.transforms.loading import LoadDataFromFile, LoadStreamDataFromFile
from tools.genie.occ_generation.scene_runtime import atomic_file, file_identity, identity_key
from tools.trajectory_generation.bev_layout import class_names_from_metainfo

logger = logging.getLogger(__name__)
ANNOTATION_CACHE_VERSION = 2


def load_scene_annotations(path, scene_name, cache_root, *, load_interval=1,
                           refresh=False, candidate_tokens=None):
    """Read each large source once; subsequent runs read only the cached scene."""
    signature = dict(version=ANNOTATION_CACHE_VERSION, source=file_identity(path),
                     scene=scene_name, load_interval=load_interval)
    if candidate_tokens is not None:
        signature['candidate_tokens'] = list(candidate_tokens)
    destination = Path(cache_root) / f'annotations_{identity_key(signature)}.pkl'
    if destination.is_file() and not refresh:
        try:
            with destination.open('rb') as stream:
                result = pickle.load(stream)
            if result['signature'] == signature:
                return result, True
        except (OSError, EOFError, ValueError, KeyError, pickle.UnpicklingError):
            logger.warning('Rebuilding invalid annotation cache: %s', destination)
    with Path(path).open('rb') as stream:
        source = pickle.load(stream)
    infos = source.get('data_list', source.get('infos'))
    if infos is None:
        raise ValueError(f'{path} must contain data_list or infos')
    metainfo = source.get('metainfo', {})
    if candidate_tokens is not None:
        # Interpolated scene names can be aliases of native nuScenes scenes.
        # Match the earliest eligible 12 Hz keyframe by token, as the old global
        # dataset lookup did; never infer native identity from the scene number.
        by_token = {str(info['token']): str(info['scene_name'])
                    for info in infos[::load_interval]}
        source_scene = next((by_token[t] for t in candidate_tokens if t in by_token), None)
        if source_scene is None:
            raise ValueError(f'No eligible 12 Hz keyframe exists in native annotations: {scene_name}')
    else:
        source_scene = scene_name
    selected, indices = [], []
    previous_scene, group = None, -1
    selected_group = None
    for index, info in enumerate(infos[::load_interval]):
        current_scene = str(info.get('scene_name'))
        if current_scene != previous_scene:
            group += 1
            previous_scene = current_scene
        if current_scene == source_scene:
            if selected_group is not None and group != selected_group:
                raise ValueError(f'{scene_name} appears in noncontiguous source segments')
            selected_group = group
            selected.append(info)
            indices.append(index)
    if not selected:
        raise ValueError(f'Scene {scene_name} was not found in {path}')
    result = dict(signature=signature, infos=selected, metainfo=metainfo,
                  source_scene=source_scene,
                  class_names=class_names_from_metainfo(metainfo),
                  source_indices=indices, sequence_group_index=selected_group)
    with atomic_file(destination) as stream:
        pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return result, False


class CachedFrameLoader(LoadDataFromFile):
    """A bounded CPU array cache shared by overlapping temporal samples."""

    def __init__(self, *args, max_entries=128, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_entries = max_entries
        self.arrays = OrderedDict()
        self.reads = self.hits = 0

    def transform(self, results):
        key = (str(self.data_root), str(results['token']), self.input_keys, self.suffix)
        if key in self.arrays:
            self.hits += 1
            self.arrays.move_to_end(key)
            results[self.output_keys] = self.arrays[key]
            return results
        results = super().transform(results)
        self.reads += 1
        if self.max_entries > 0:
            value = results[self.output_keys]
            value.setflags(write=False)
            self.arrays[key] = value
            if len(self.arrays) > self.max_entries:
                self.arrays.popitem(last=False)
        return results


class CachedStreamData(LoadStreamDataFromFile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.load_data_from_file = CachedFrameLoader(
            data_root=self.data_root, input_keys=self.input_keys,
            output_keys=self.output_keys, suffix=self.suffix)


class SceneInferenceDataset(NuScenesWorldDataset):
    def __init__(self, scene_annotations, **kwargs):
        self.scene_annotations = scene_annotations
        super().__init__(**kwargs)

    def load_data_list(self):
        for key, value in self.scene_annotations['metainfo'].items():
            self._metainfo.setdefault(key, value)
        return [self.parse_data_info(copy.deepcopy(info))
                for info in self.scene_annotations['infos']]

    def get_data_info(self, idx):
        info = super().get_data_info(idx)
        source_index = self.scene_annotations['source_indices'][idx]
        info['sample_idx'] = source_index
        info['sample_index'] = source_index
        info['sequence_group_idx'] = self.scene_annotations['sequence_group_index']
        return info


def build_scene_dataset(cfg, annotations, layout_root):
    config = copy.deepcopy(cfg.val_dataloader.dataset)
    if config.pop('type') != 'NuScenesWorldDataset':
        raise ValueError('Scene inference currently requires NuScenesWorldDataset')
    if config.get('indices') is not None or config.get('filter_cfg'):
        raise ValueError('Scene inference requires an unfiltered native dataset')
    pipeline = []
    for transform in config['pipeline']:
        transform = copy.deepcopy(transform)
        if transform.get('output_keys') in ('bev_latent', 'bev_layout'):
            transform['type'] = CachedStreamData
            if transform['output_keys'] == 'bev_layout':
                transform['data_root'] = str(layout_root)
            pipeline.append(transform)
        elif transform['type'] == 'MMPack3DDetInputs':
            transform['keys'] = ['bev_latent', 'bev_layout']
            pipeline.append(transform)
    if len(pipeline) != 3:
        raise ValueError('Expected latent/layout loaders and one pack transform')
    config.update(pipeline=pipeline, serialize_data=False, load_interval=1,
                  load_futu_number=0, test_mode=True, use_sequence_group_flag=True,
                  sequences_split_num=1)
    return SceneInferenceDataset(annotations, **config)


def latent_loader(dataset):
    for transform in dataset.pipeline.transforms:
        if getattr(transform, 'output_keys', None) == 'bev_latent':
            return transform.load_data_from_file
    raise ValueError('Missing latent loader')


@torch.no_grad()
def warmup_scene_memo(model, dataset, target_index, warmup_frames):
    """Read only the current latent of each warmup sample, in original order."""
    model.memo_bev_latent = None
    loader = latent_loader(dataset)
    warmed = []
    device = next(model.parameters()).device
    for index in range(max(0, target_index - warmup_frames), target_index):
        info = dataset.get_data_info(index)
        array = loader.transform(dict(token=info['token']))['bev_latent']
        # A copy avoids exposing the read-only CPU cache to model mutations.
        current = torch.tensor(array, dtype=torch.float32, device=device)[None, None]
        model.get_memo_latent(current, [info['start_of_sequence']])
        model.update_memo_latent(current)
        warmed.append(str(info['token']))
    return warmed


def input_cache_stats(dataset):
    return {t.output_keys: dict(reads=t.load_data_from_file.reads,
                                hits=t.load_data_from_file.hits)
            for t in dataset.pipeline.transforms if isinstance(t, CachedStreamData)}


def cached_layout_arrays(path, signature, compute, frames, *, refresh=False):
    """Cache a complete future layout window, validating shape and provenance."""
    key = identity_key(signature)
    path = Path(path)
    shapes = dict(layouts=(frames, 18, 200, 200), transforms=(frames, 4, 4))
    if path.is_file() and not refresh:
        try:
            with np.load(path, allow_pickle=False) as archive:
                valid = str(archive['signature']) == key
                values = {name: archive[name] for name in shapes}
            valid = valid and all(values[k].shape == v for k, v in shapes.items())
            valid = valid and values['layouts'].dtype == np.uint8
            valid = valid and np.isfinite(values['transforms']).all()
            if valid:
                return values['layouts'], values['transforms'], True
        except (OSError, ValueError, KeyError, EOFError):
            pass
        logger.info('Future layout cache invalidated: %s', path)
    layouts, transforms = compute()
    if layouts.shape != shapes['layouts'] or transforms.shape != shapes['transforms']:
        raise ValueError('Unexpected future layout or transform shape')
    if layouts.dtype != np.uint8 or not np.isfinite(transforms).all():
        raise ValueError('Future layouts must be uint8 and transforms must be finite')
    with atomic_file(path) as stream:
        np.savez_compressed(stream, signature=np.asarray(key),
                            layouts=layouts, transforms=transforms)
    return layouts, transforms, False
