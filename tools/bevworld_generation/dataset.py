"""Input preprocessing and temporal-memory helpers."""

from __future__ import annotations

import torch
from mmengine.dataset import pseudo_collate


def scene_start_index(dataset, global_index):
    info = dataset.get_data_info(global_index)
    scene_name = info["scene_name"]
    index = global_index
    while index > 0:
        if dataset.get_data_info(index - 1).get("scene_name") != scene_name:
            break
        index -= 1
    return index


def prepare_batch(model, dataset, index):
    data_batch = pseudo_collate([dataset[index]])
    return model.data_preprocessor(data_batch, training=False)


def current_latent_and_memo_inputs(model, data):
    batch_inputs = data["inputs"]
    batch_data_samples = data["data_samples"]
    bev_latent = batch_inputs["bev_latent"]
    current = bev_latent[:, model.prev_number:model.prev_number + 1]
    starts = [
        sample.metainfo["start_of_sequence"]
        for sample in batch_data_samples
    ]
    return batch_inputs, batch_data_samples, current, starts


@torch.no_grad()
def warmup_memo(model, dataset, target_index, warmup_frames):
    if warmup_frames <= 0:
        model.memo_bev_latent = None
        return []
    start_index = max(
        scene_start_index(dataset, target_index), target_index - warmup_frames)
    warmed = []
    model.memo_bev_latent = None
    for index in range(start_index, target_index):
        data = prepare_batch(model, dataset, index)
        _, samples, current, starts = current_latent_and_memo_inputs(model, data)
        model.get_memo_latent(current, starts)
        model.update_memo_latent(current)
        warmed.append(samples[0].metainfo.get("token"))
    return warmed
