from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention

class LayerScale(nn.Module):
    """Layer scale module.

    References:
      - https://arxiv.org/abs/2103.17239
    """

    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        """Initialize LayerScale module.

        Args:
            dim: Dimension.
            init_values: Initial value for scaling.
            inplace: If True, perform inplace operations.
        """
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer scaling."""
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

from timm.layers import (
    Mlp,
)

class FutureContextDeformableFlowBlock(nn.Module):
    """Flow block that updates future tokens by querying encoded context."""

    def __init__(self,
                 dim,
                 num_heads,
                 context_frames,
                 future_frames,
                 seq_h,
                 seq_w,
                 spatial_num_levels=1,
                 spatial_num_points=4,
                 context_num_points=4,
                 use_future_temporal_attn=False,
                 future_temporal_num_points=4,
                 future_temporal_init_value=1e-3,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 proj_drop=0.0,
                 attn_drop=0.0):
        super().__init__()
        del qkv_bias
        self.context_frames = context_frames
        self.future_frames = future_frames
        self.seq_h = seq_h
        self.seq_w = seq_w
        self.seq_len = seq_h * seq_w
        self.use_future_temporal_attn = bool(use_future_temporal_attn)

        self.norm_spatial = norm_layer(dim)
        self.attn_spatial = MultiScaleDeformableAttention(
            embed_dims=dim,
            num_heads=num_heads,
            num_levels=spatial_num_levels,
            num_points=spatial_num_points,
            dropout=attn_drop,
            batch_first=True,
        )

        self.norm_future_query = norm_layer(dim)
        self.norm_context_value = norm_layer(dim)
        self.attn_context = MultiScaleDeformableAttention(
            embed_dims=dim,
            num_heads=num_heads,
            num_levels=context_frames,
            num_points=context_num_points,
            dropout=attn_drop,
            batch_first=True,
        )

        if self.use_future_temporal_attn:
            self.norm_future_temporal = norm_layer(dim)
            self.attn_future_temporal = MultiScaleDeformableAttention(
                embed_dims=dim,
                num_heads=num_heads,
                num_levels=future_frames,
                num_points=future_temporal_num_points,
                dropout=attn_drop,
                batch_first=True,
            )
            self.future_temporal_scale = LayerScale(
                dim, init_values=future_temporal_init_value)
            self.register_buffer(
                'future_spatial_shapes',
                torch.tensor(
                    [[seq_h, seq_w]] * future_frames, dtype=torch.long),
                persistent=False)
            self.register_buffer(
                'future_level_start_index',
                torch.arange(future_frames, dtype=torch.long) * self.seq_len,
                persistent=False)
        else:
            self.norm_future_temporal = None
            self.attn_future_temporal = None
            self.future_temporal_scale = None

        self.norm_mlp = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=nn.GELU,
            bias=True,
            drop=proj_drop,
        )

        self.register_buffer(
            'context_spatial_shapes',
            torch.tensor([[seq_h, seq_w]] * context_frames, dtype=torch.long),
            persistent=False)
        self.register_buffer(
            'context_level_start_index',
            torch.arange(context_frames, dtype=torch.long) * self.seq_len,
            persistent=False)

    def _context_reference_points(self, batch_size, device, dtype):
        y_embed, x_embed = torch.meshgrid(
            torch.arange(self.seq_h, device=device, dtype=dtype),
            torch.arange(self.seq_w, device=device, dtype=dtype),
            indexing='ij')
        ref_y = (y_embed.reshape(-1) + 0.5) / self.seq_h
        ref_x = (x_embed.reshape(-1) + 0.5) / self.seq_w
        spatial_reference = torch.stack((ref_x, ref_y), dim=-1)
        spatial_reference = spatial_reference.view(1, 1, self.seq_len, 1, 2)
        reference_points = spatial_reference.expand(
            batch_size, self.future_frames, -1, self.context_frames, -1)
        return reference_points.reshape(
            batch_size, self.future_frames * self.seq_len,
            self.context_frames, 2)

    def _future_reference_points(self, batch_size, device, dtype):
        y_embed, x_embed = torch.meshgrid(
            torch.arange(self.seq_h, device=device, dtype=dtype),
            torch.arange(self.seq_w, device=device, dtype=dtype),
            indexing='ij')
        ref_y = (y_embed.reshape(-1) + 0.5) / self.seq_h
        ref_x = (x_embed.reshape(-1) + 0.5) / self.seq_w
        spatial_reference = torch.stack((ref_x, ref_y), dim=-1)
        spatial_reference = spatial_reference.view(1, 1, self.seq_len, 1, 2)
        reference_points = spatial_reference.expand(
            batch_size, self.future_frames, -1, self.future_frames, -1)
        return reference_points.reshape(
            batch_size, self.future_frames * self.seq_len,
            self.future_frames, 2)

    def forward(self,
                future,
                context,
                reference_points,
                spatial_shapes,
                level_start_index,
                key_padding_mask=None):
        batch_size, future_frames, seq_len, channels = future.shape
        context_batch, context_frames, context_seq_len, context_channels = \
            context.shape
        if future_frames != self.future_frames or seq_len != self.seq_len:
            raise ValueError(
                f'Expected future shape (B, {self.future_frames}, '
                f'{self.seq_len}, C), got {tuple(future.shape)}.')
        if (context_batch != batch_size or
                context_frames != self.context_frames or
                context_seq_len != self.seq_len or
                context_channels != channels):
            raise ValueError(
                f'Expected context shape (B, {self.context_frames}, '
                f'{self.seq_len}, {channels}), got {tuple(context.shape)}.')

        future_spatial = future.reshape(batch_size * future_frames, seq_len,
                                        channels)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.reshape(
                batch_size * future_frames, seq_len)

        spatial_out = self.attn_spatial(
            query=self.norm_spatial(future_spatial),
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=key_padding_mask)
        x = future_spatial + spatial_out
        x = x.reshape(batch_size, future_frames, seq_len, channels)

        future_query = x.reshape(batch_size, future_frames * seq_len, channels)
        context_value = context.reshape(
            batch_size, context_frames * seq_len, channels)
        context_reference_points = self._context_reference_points(
            batch_size, x.device, x.dtype)
        context_out = self.attn_context(
            query=self.norm_future_query(future_query),
            value=self.norm_context_value(context_value),
            identity=future_query,
            reference_points=context_reference_points,
            spatial_shapes=self.context_spatial_shapes.to(device=x.device),
            level_start_index=self.context_level_start_index.to(
                device=x.device))
        future_query = context_out

        if self.attn_future_temporal is not None:
            temporal_value = self.norm_future_temporal(future_query)
            temporal_reference_points = self._future_reference_points(
                batch_size, x.device, x.dtype)
            temporal_key_padding_mask = (
                None if key_padding_mask is None else
                key_padding_mask.reshape(batch_size, future_frames * seq_len))
            temporal_out = self.attn_future_temporal(
                query=temporal_value,
                value=temporal_value,
                identity=torch.zeros_like(future_query),
                reference_points=temporal_reference_points,
                spatial_shapes=self.future_spatial_shapes.to(device=x.device),
                level_start_index=self.future_level_start_index.to(
                    device=x.device),
                key_padding_mask=temporal_key_padding_mask)
            future_query = future_query + self.future_temporal_scale(
                temporal_out)

        x = future_query.reshape(
            batch_size, future_frames, seq_len, channels)
        return x + self.mlp(self.norm_mlp(x))
