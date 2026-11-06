"""Minimal LO-World rollout helpers for Genie OCC condition generation."""

from __future__ import annotations

import time

import numpy as np
import torch
from einops import rearrange

def build_layout_sequence(model, dataset_layout, future_layout):
    """Append 12 Hz future layouts after the model history/current frames."""
    if dataset_layout.dim() != 5 or dataset_layout.size(0) != 1:
        raise ValueError(
            f"Expected dataset layout [1,T,C,X,Y], got "
            f"{tuple(dataset_layout.shape)}.")
    future = torch.as_tensor(
        future_layout,
        device=dataset_layout.device,
        dtype=dataset_layout.dtype).unsqueeze(0)
    if tuple(future.shape[2:]) != tuple(dataset_layout.shape[2:]):
        raise ValueError(
            f"Future layout spatial shape {tuple(future.shape[2:])} differs "
            f"from dataset shape {tuple(dataset_layout.shape[2:])}.")
    prefix = dataset_layout[:, :model.prev_number + 1]
    return torch.cat([prefix, future], dim=1)


def reset_generation_seed(seed):
    """Reset CPU and CUDA random generators for reproducible rollout."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def rollout(
        model, current_latent, memo_latent, ego, layout, frames, seed,
        progress_callback=None):
    """Generate the requested number of future BEV latent frames."""
    reset_generation_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(current_latent.device)
        torch.cuda.synchronize(current_latent.device)
    start = time.perf_counter()
    prediction = model.rollout_future_latents(
        current_latent.clone(),
        memo_latent.clone(),
        ego,
        future_number=frames,
        rollout_stride=model.futu_number,
        bev_layout=layout,
        progress_callback=progress_callback)
    if torch.cuda.is_available():
        torch.cuda.synchronize(current_latent.device)
        peak_mb = (
            torch.cuda.max_memory_allocated(current_latent.device) / 2**20)
    else:
        peak_mb = 0.0
    return prediction, time.perf_counter() - start, peak_mb


@torch.no_grad()
def decode_occ(model, latents, chunk_size):
    """Decode occupancy without requiring range-image metadata."""
    outputs = []
    shapes = model._vae_decoder_shapes(latents.device)
    for start in range(0, latents.size(1), chunk_size):
        chunk = latents[:, start:start + chunk_size]
        flat = rearrange(chunk, "b t c h w -> (b t) c h w")
        logits = model.vae.decode_scene_logits(flat, shapes)
        labels = logits.argmax(dim=-1).to(torch.uint8).cpu().numpy()
        outputs.append(labels)
        del logits, labels, flat
    return np.concatenate(outputs, axis=0)
