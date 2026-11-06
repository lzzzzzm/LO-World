import numpy as np

from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS
from mmdet3d.models.detectors import MVXTwoStageDetector
from mmdet3d.structures import Det3DDataSample

from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)

from projects.models.tokenizer.occ_lidar_vae import projected_points
from projects.models.world_models.condition_net import LayoutNet, MotionNet
from projects.models.world_models.custom_block import (
    FutureContextDeformableFlowBlock,
)


@MODELS.register_module()
class BEVWorld(MVXTwoStageDetector):
    """BEV latent world model baseline with sequence memory and flow matching."""

    def __init__(self,
                 vae=None,
                 vae_pretrained=None,
                 memo_number=4,
                 prev_number=0,
                 futu_number=6,
                 img_size=(200, 200),
                 vae_stride=4,
                 vae_embed_dim=16,
                 encoder_embed_dim=1024,
                 encoder_depth=16,
                 encoder_num_heads=16,
                 flow_depth=4,
                 use_future_temporal_attn=False,
                 future_temporal_num_points=4,
                 future_temporal_init_value=1e-3,
                 mlp_ratio=4.0,
                 patch_size=1,
                 norm_layer=nn.LayerNorm,
                 proj_dropout=0.1,
                 attn_dropout=0.1,
                 flow_num_train_timesteps=1000,
                 flow_num_steps=20,
                 flow_shift=1.0,
                 flow_inference_schedule='default',
                 flow_inference_lq_threshold=0.025,
                 flow_inference_lq_linear_steps=None,
                 use_bev_layout=False,
                 bev_layout_in_channels=18,
                 world_pretrained=None,
                 rollout_future_number=None,
                 rollout_stride=None,
                 train_autoregressive_chunks=1,
                 train_autoregressive_loss_weight=1.0,
                 train_autoregressive_chunk_mode='all',
                 range_lidar_loss_weight=0.0,
                 range_lidar_loss_frame_indices=None,
                 future_padding_loss_weight=1.0,
                 decode_chunk_size=None,
                 preserve_full_pred_points=False,
                 use_projected_gt=False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.decode_chunk_size = decode_chunk_size
        self.preserve_full_pred_points = bool(preserve_full_pred_points)
        self.use_projected_gt = bool(use_projected_gt)

        # ------------- VAE -------------
        self.vae = MODELS.build(vae)
        self._load_vae_weights(vae_pretrained)
        for param in self.vae.parameters():
            param.requires_grad = False

        token_embed_dim = vae_embed_dim * patch_size**2
        seq_h = img_size[0] // vae_stride // patch_size
        seq_w = img_size[1] // vae_stride // patch_size
        seq_len = seq_h * seq_w
        context_frames = memo_number + 1
        # ------------ Projectors ------------
        self.cond_proj = nn.Linear(token_embed_dim, encoder_embed_dim, bias=True)
        self.noisy_proj = nn.Linear(token_embed_dim, encoder_embed_dim, bias=True)
        self.velocity_proj = nn.Linear(encoder_embed_dim, token_embed_dim, bias=True)

        # ------------ Position / condition embeddings ------------
        self.context_temporal_pos_embed = nn.Parameter(
            torch.zeros(1, context_frames, encoder_embed_dim))
        self.future_temporal_pos_embed = nn.Parameter(
            torch.zeros(1, futu_number, encoder_embed_dim))
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, seq_len, encoder_embed_dim))
        self.time_embed = nn.Sequential(
            nn.Linear(1, encoder_embed_dim),
            nn.SiLU(),
            nn.Linear(encoder_embed_dim, encoder_embed_dim),
        )
        self.motion_embed = MotionNet(embed_dims=encoder_embed_dim)
        self.use_bev_layout = bool(use_bev_layout)
        self.bev_layout_in_channels = int(bev_layout_in_channels)
        if self.use_bev_layout:
            self.bev_layout_embed = LayoutNet(
                in_channel=self.bev_layout_in_channels,
                embed_dims=encoder_embed_dim)
            self.bev_layout_adapter = self._build_bev_layout_adapter(
                encoder_embed_dim)
        else:
            self.bev_layout_embed = None
            self.bev_layout_adapter = None

        # ------------ Context encoder and flow denoiser ------------
        self.context_blocks = nn.ModuleList([
            FutureContextDeformableFlowBlock(
                    dim=encoder_embed_dim,
                    num_heads=encoder_num_heads,
                    context_frames=context_frames,
                    future_frames=context_frames,
                    seq_h=seq_h,
                    seq_w=seq_w,
                    spatial_num_levels=1,
                    spatial_num_points=4,
                    context_num_points=4,
                    use_future_temporal_attn=False,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    proj_drop=proj_dropout,
                    attn_drop=attn_dropout)
            for _ in range(encoder_depth)
        ])
        self.context_norm = norm_layer(encoder_embed_dim)

        flow_block = FutureContextDeformableFlowBlock
        self.flow_blocks = nn.ModuleList()
        for _ in range(flow_depth):
            self.flow_blocks.append(
                flow_block(
                    dim=encoder_embed_dim,
                    num_heads=encoder_num_heads,
                    context_frames=context_frames,
                    future_frames=futu_number,
                    seq_h=seq_h,
                    seq_w=seq_w,
                    spatial_num_levels=1,
                    spatial_num_points=4,
                    context_num_points=4,
                    use_future_temporal_attn=use_future_temporal_attn,
                    future_temporal_num_points=future_temporal_num_points,
                    future_temporal_init_value=future_temporal_init_value,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    proj_drop=proj_dropout,
                    attn_drop=attn_dropout,
                ))
        self.flow_norm = norm_layer(encoder_embed_dim)

        # ------------ Parameters ------------
        self.memo_number = memo_number
        self.prev_number = prev_number
        self.futu_number = futu_number
        self.seq_h = seq_h
        self.seq_w = seq_w
        self.seq_len = seq_len
        self.token_embed_dim = token_embed_dim
        self.encoder_embed_dim = encoder_embed_dim
        self.patch_size = patch_size
        self.flow_num_steps = flow_num_steps
        self.flow_num_train_timesteps = flow_num_train_timesteps
        self.flow_shift = float(flow_shift)
        self.flow_inference_schedule = flow_inference_schedule
        self.flow_inference_lq_threshold = float(flow_inference_lq_threshold)
        self.flow_inference_lq_linear_steps = flow_inference_lq_linear_steps
        self.flow_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=flow_num_train_timesteps,
            shift=flow_shift)
        self.rollout_future_number = (
            None if rollout_future_number is None
            else int(rollout_future_number))
        self.rollout_stride = (
            self.futu_number if rollout_stride is None else int(rollout_stride))
        self.train_autoregressive_chunks = int(train_autoregressive_chunks)
        self.train_autoregressive_loss_weight = float(
            train_autoregressive_loss_weight)
        self.train_autoregressive_chunk_mode = (
            train_autoregressive_chunk_mode)
        self._train_autoregressive_chunk_cursor = 0
        self.range_lidar_loss_weight = float(range_lidar_loss_weight)
        self.range_lidar_loss_frame_indices = (
            None if range_lidar_loss_frame_indices is None else
            [int(index) for index in range_lidar_loss_frame_indices])
        self.future_padding_loss_weight = float(future_padding_loss_weight)

        self.memo_bev_latent = None

        self.initialize_weights()
        self._reset_bev_layout_adapter()
        self.world_pretrained = world_pretrained
        if self.world_pretrained is not None:
            self._load_world_weights(self.world_pretrained)

    def _load_vae_weights(self, checkpoint_path):
        """Initialize the tokenizer unless a full World checkpoint will restore it."""
        if checkpoint_path is None:
            return
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.vae.load_state_dict(checkpoint['state_dict'], strict=True)

    def _adapt_future_temporal_pos_embed(self, value):
        target = self.future_temporal_pos_embed.data
        if value.shape == target.shape:
            return value
        if (value.dim() != 3 or target.dim() != 3 or
                value.size(0) != target.size(0) or
                value.size(2) != target.size(2)):
            return None
        adapted = target.clone()
        num_frames = min(value.size(1), target.size(1))
        adapted[:, :num_frames] = value[:, :num_frames]
        return adapted

    def _adapt_future_temporal_attn_parameter(self, key, value):
        """Expand a temporal deformable-attention parameter to a new horizon."""
        supported = {
            'sampling_offsets.weight': 2,
            'sampling_offsets.bias': 2,
            'attention_weights.weight': 1,
            'attention_weights.bias': 1,
        }
        suffix = next(
            (name for name in supported if key.endswith(name)), None)
        if suffix is None or '.attn_future_temporal.' not in key:
            return None

        module_path = key.rsplit('.', 2)[0]
        attention = self.get_submodule(module_path)
        target = self.state_dict()[key]
        coord_dims = supported[suffix]
        divisor = attention.num_heads * attention.num_points * coord_dims
        if value.shape[0] % divisor != 0 or target.shape[0] % divisor != 0:
            return None

        old_levels = value.shape[0] // divisor
        new_levels = target.shape[0] // divisor
        old_shape = (
            attention.num_heads, old_levels, attention.num_points,
            coord_dims, *value.shape[1:])
        new_shape = (
            attention.num_heads, new_levels, attention.num_points,
            coord_dims, *target.shape[1:])
        adapted = target.clone().reshape(new_shape)
        source = value.reshape(old_shape)
        copied_levels = min(old_levels, new_levels)
        adapted[:, :copied_levels] = source[:, :copied_levels]
        return adapted.reshape_as(target)

    def _load_world_weights(
            self,
            checkpoint_path,
            allowed_missing_prefixes=(
                'vae.',
                'bev_layout_embed.',
                'bev_layout_adapter.')):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        current_state = self.state_dict()
        filtered_state = {}
        skipped_keys = []

        for key, value in state_dict.items():
            key = key[7:] if key.startswith('module.') else key
            if key.startswith('vae.'):
                continue
            if key not in current_state:
                skipped_keys.append(key)
                continue
            if current_state[key].shape != value.shape:
                if key == 'future_temporal_pos_embed':
                    value = self._adapt_future_temporal_pos_embed(value)
                    if value is not None:
                        filtered_state[key] = value
                        continue
                value = self._adapt_future_temporal_attn_parameter(
                    key, value)
                if value is not None:
                    filtered_state[key] = value
                    continue
                skipped_keys.append(key)
                continue
            filtered_state[key] = value

        incompatible = self.load_state_dict(filtered_state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = [
            key for key in incompatible.missing_keys
            if not any(
                key.startswith(prefix)
                for prefix in allowed_missing_prefixes)
        ]
        fatal_skipped = [
            key for key in skipped_keys
            if key in current_state and not any(
                key.startswith(prefix)
                for prefix in allowed_missing_prefixes)
        ]
        if missing or unexpected or fatal_skipped:
            raise RuntimeError(
                'Failed to load BEVWorld checkpoint cleanly. '
                f'Missing non-VAE keys: {missing[:10]}, '
                f'unexpected keys: {unexpected[:10]}, '
                f'skipped keys: {fatal_skipped[:10]}')

    def initialize_weights(self):
        torch.nn.init.normal_(self.context_temporal_pos_embed, std=0.02)
        torch.nn.init.normal_(self.future_temporal_pos_embed, std=0.02)
        torch.nn.init.normal_(self.spatial_pos_embed, std=0.02)

        for name, module in self.named_modules():
            if name.startswith('vae'):
                continue
            self._init_weights(module)

    @staticmethod
    def _build_bev_layout_adapter(embed_dim):
        adapter_in_dim = embed_dim * 2
        adapter = nn.Sequential(
            nn.LayerNorm(adapter_in_dim),
            nn.Linear(adapter_in_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        nn.init.zeros_(adapter[-1].weight)
        nn.init.zeros_(adapter[-1].bias)
        return adapter

    def _reset_bev_layout_adapter(self):
        if self.bev_layout_adapter is not None:
            nn.init.zeros_(self.bev_layout_adapter[-1].weight)
            nn.init.zeros_(self.bev_layout_adapter[-1].bias)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
            if module.weight is not None:
                nn.init.constant_(module.weight, 1.0)

    def _expand_decode_data_samples(self, batch_data_samples,
                                    frames_per_sample, start_frame=None):
        if start_frame is None:
            start_frame = self.prev_number
        return self._select_decode_data_samples(
            batch_data_samples,
            range(start_frame, start_frame + frames_per_sample))

    def _select_decode_data_samples(self, batch_data_samples, frame_indices):
        """Attach each decoded frame's real LiDAR extrinsic and token."""
        decode_samples = []
        for data_sample in batch_data_samples:
            metainfo = data_sample.metainfo
            transforms = np.asarray(metainfo['total_lidar2ego'])
            for frame_idx in frame_indices:
                lidar2ego = transforms[frame_idx]
                frame_sample = Det3DDataSample()
                frame_sample.set_metainfo(dict(
                    lidar2ego=lidar2ego,
                    ego2lidar=np.linalg.inv(lidar2ego),
                    token=metainfo['total_token'][frame_idx]))
                decode_samples.append(frame_sample)
        return decode_samples

    def obtain_data_from_latent(self, bev_latent, batch_data_samples=None,
                                frames_per_sample=1, start_frame=None):
        """Decode frames using metadata starting at the current frame by default.

        Future-only callers use ``start_frame=self.prev_number + 1``.
        """
        expanded = self._expand_decode_data_samples(
            batch_data_samples, frames_per_sample, start_frame=start_frame)
        if self.decode_chunk_size is None:
            return self._decode_latent_batch(bev_latent, expanded)
        occ_frames, points = [], []
        for start in range(0, len(bev_latent), self.decode_chunk_size):
            end = start + self.decode_chunk_size
            occ, frame_points = self._decode_latent_batch(
                bev_latent[start:end], expanded[start:end])
            occ_frames.append(occ)
            points.extend(frame_points)
        return torch.cat(occ_frames, dim=0), points

    @torch.no_grad()
    def _decode_latent_batch(self, bev_latent, decode_samples):
        """Decode joint OCC semantics and LiDAR points with OccLiDARVAE."""
        occ_logits, depths = self.vae.decode_joint(
            bev_latent, self._vae_decoder_shapes(bev_latent.device),
            decode_samples)
        return occ_logits.argmax(dim=-1), self.vae.depths_to_points(depths)

    @staticmethod
    def _range_depths(batch_inputs_dict):
        """Read channel-last range images from cached or on-the-fly pipelines."""
        images = batch_inputs_dict['range_imgs']
        # Cached NuScenes frames include a singleton dimension before H,W,C;
        # Waymo's on-the-fly projection already stores [B,T,H,W,C].
        return images.reshape(*images.shape[:2], *images.shape[-3:])[..., 0].float()

    def _target_points_from_range(self, batch_inputs_dict, frames_per_sample):
        depths = self._range_depths(batch_inputs_dict)[
            :, self.prev_number:self.prev_number + frames_per_sample]
        return [
            [self.vae._range_depth_to_points_lidar(depth) for depth in sample]
            for sample in depths
        ]

    def _range_targets_from_batch(self, batch_inputs_dict, frame_indices):
        depths = self._range_depths(batch_inputs_dict)[:, frame_indices]
        return depths.flatten(0, 1).unsqueeze(1)

    @staticmethod
    def _vae_decoder_shapes(device):
        return [
            torch.tensor((200, 200), device=device),
            torch.tensor((100, 100), device=device),
        ]

    def _predict_clean_tokens(self, noisy_tokens, pred_velocity, sigmas):
        while sigmas.dim() < noisy_tokens.dim():
            sigmas = sigmas.unsqueeze(-1)
        return noisy_tokens - sigmas * pred_velocity

    def _range_lidar_frame_indices(self):
        if self.range_lidar_loss_frame_indices is not None:
            return self.range_lidar_loss_frame_indices
        return list(range(self.futu_number))

    def _future_padding_frame_weights(self, batch_data_samples, future_offset,
                                      device, dtype, frame_indices=None):
        if self.future_padding_loss_weight == 1.0:
            return None
        if frame_indices is None:
            frame_indices = range(self.futu_number)
        indices = torch.tensor(
            [future_offset + index for index in frame_indices],
            device=device, dtype=torch.long)
        masks = torch.stack([
            torch.as_tensor(sample.metainfo['futu_padding_mask'],
                            device=device, dtype=torch.bool)[indices]
            for sample in batch_data_samples
        ])
        return torch.where(masks, self.future_padding_loss_weight, 1.0).to(dtype)

    def _compute_range_lidar_loss(self, range_logits, range_imgs,
                                  sample_frame_weights=None):
        vae = self.vae
        edge_loss_weight = vae.range_edge_loss_weight
        depth_loss_weight = vae.range_depth_loss_weight

        target_bins, valid = vae._prepare_range_target(range_imgs)
        range_ce = F.cross_entropy(
            range_logits.float(), target_bins, reduction='none')
        pixel_weights = torch.where(
            valid,
            range_ce.new_full(range_ce.shape,
                              vae.valid_pixel_loss_weight),
            range_ce.new_full(range_ce.shape,
                              vae.empty_pixel_loss_weight))
        edge_weights = vae._range_edge_weights(range_imgs, valid)
        if edge_loss_weight > 0 and valid.any():
            edge_weights = torch.where(
                valid,
                edge_weights / edge_weights[valid].mean().clamp(min=1e-6),
                torch.ones_like(edge_weights))
        pixel_weights = pixel_weights * edge_weights

        frame_weights = None
        if sample_frame_weights is not None:
            frame_weights = sample_frame_weights.to(
                device=range_logits.device,
                dtype=range_logits.dtype).reshape(-1, 1, 1)
        if frame_weights is not None:
            pixel_weights = pixel_weights * frame_weights

        loss_range = (range_ce * pixel_weights).mean()
        losses = dict(
            loss_range_lidar_ce=(
                loss_range * self.range_lidar_loss_weight))

        if depth_loss_weight > 0:
            target_depth = vae.lidar_utils.convert_depth(
                range_imgs[:, 0])
            target_depth = vae.lidar_utils.normalize(target_depth)
            pred_depth = vae._range_logits_to_soft_normalized_depth(
                range_logits)
            if valid.any():
                depth_reg = F.smooth_l1_loss(
                    pred_depth, target_depth, reduction='none')
                depth_weights = edge_weights
                if frame_weights is not None:
                    depth_weights = depth_weights * frame_weights
                depth_loss = (depth_reg[valid] *
                              depth_weights[valid]).mean()
            else:
                depth_loss = pred_depth.sum() * 0.0
            losses['loss_range_lidar_depth'] = (
                depth_loss * depth_loss_weight *
                self.range_lidar_loss_weight)
        return losses

    def _range_lidar_aux_losses(self, pred_velocity, noisy_tokens, sigmas,
                                batch_inputs_dict, batch_data_samples,
                                future_offset=0, loss_suffix='',
                                loss_weight=1.0):
        if self.range_lidar_loss_weight <= 0:
            return {}

        relative_indices = self._range_lidar_frame_indices()
        absolute_indices = [
            self.prev_number + 1 + future_offset + index
            for index in relative_indices
        ]
        range_imgs = self._range_targets_from_batch(
            batch_inputs_dict, absolute_indices)

        clean_tokens = self._predict_clean_tokens(
            noisy_tokens, pred_velocity, sigmas)
        selected_tokens = clean_tokens[:, relative_indices]
        selected_tokens = rearrange(selected_tokens, 'b t s c -> (b t) s c')
        pred_latent = self.unpatchify(selected_tokens)

        decode_samples = self._select_decode_data_samples(
            batch_data_samples, absolute_indices)
        range_shapes = self._vae_decoder_shapes(pred_latent.device)
        _, range_logits = self.vae.decode_joint_with_logits(
            pred_latent, occ_shapes=range_shapes, data_samples=decode_samples)

        losses = self._compute_range_lidar_loss(
            range_logits, range_imgs.to(pred_latent.device),
            sample_frame_weights=self._future_padding_frame_weights(
                batch_data_samples,
                future_offset,
                pred_latent.device,
                pred_latent.dtype,
                frame_indices=relative_indices))
        suffix = f'_{loss_suffix}' if loss_suffix else ''
        return {
            f'{name}{suffix}': value * loss_weight
            for name, value in losses.items()
        }

    def patchify(self, x):
        bsz, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum("nchpwq->nhwcpq", x)
        x = x.reshape(bsz, h_ * w_, c * p**2)
        return x

    def unpatchify(self, x):
        bsz = x.shape[0]
        p = self.patch_size
        c = x.shape[-1] // (p**2)
        h_, w_ = self.seq_h, self.seq_w

        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum("nhwcpq->nchpwq", x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x

    def _patchify_sequence(self, bev_latent):
        bsz = bev_latent.size(0)
        bev_latent = rearrange(bev_latent, 'b t c h w -> (b t) c h w')
        tokens = self.patchify(bev_latent)
        return rearrange(tokens, '(b t) s c -> b t s c', b=bsz)

    def get_memo_latent(self, curr_bev_latent, start_of_sequence):
        if (self.memo_bev_latent is None or
                self.memo_bev_latent.size(0) != curr_bev_latent.size(0)):
            self.memo_bev_latent = curr_bev_latent.detach().repeat(
                1, self.memo_number, 1, 1, 1).clone()

        for index, sos in enumerate(start_of_sequence):
            if sos:
                self.memo_bev_latent[index] = curr_bev_latent[
                    index].detach().repeat(self.memo_number, 1, 1, 1)
        return self.memo_bev_latent.detach()

    def update_memo_latent(self, curr_bev_latent):
        if self.memo_number <= 0:
            return
        with torch.no_grad():
            curr = curr_bev_latent.detach()
            if self.memo_bev_latent is None:
                self.memo_bev_latent = curr.repeat(1, self.memo_number, 1, 1, 1)
            else:
                self.memo_bev_latent = torch.cat(
                    [self.memo_bev_latent[:, 1:], curr], dim=1)

    def _append_memo_latent(self, memo_bev_latent, new_bev_latent):
        if self.memo_number <= 0:
            return memo_bev_latent
        memo_bev_latent = torch.cat(
            [memo_bev_latent, new_bev_latent.detach()], dim=1)
        return memo_bev_latent[:, -self.memo_number:]

    def _build_reference_points(self, batch_size, num_frames, device, dtype):
        spatial_shapes = torch.tensor(
            [[self.seq_h, self.seq_w]], device=device, dtype=torch.long)
        level_start_index = torch.tensor([0], device=device, dtype=torch.long)
        y_embed, x_embed = torch.meshgrid(
            torch.arange(self.seq_h, device=device, dtype=dtype),
            torch.arange(self.seq_w, device=device, dtype=dtype),
            indexing='ij')
        ref_y = (y_embed.reshape(-1) + 0.5) / self.seq_h
        ref_x = (x_embed.reshape(-1) + 0.5) / self.seq_w
        reference_points = torch.stack((ref_x, ref_y), dim=-1)
        reference_points = reference_points.unsqueeze(0).unsqueeze(2)
        reference_points = reference_points.repeat(
            batch_size * num_frames, 1, 1, 1)
        return reference_points, spatial_shapes, level_start_index

    def _motion_embedding(self, curr2futu_ego):
        bsz = curr2futu_ego.size(0)
        motion_rotation = curr2futu_ego[:, :, :3, :3].reshape(
            bsz, self.futu_number, -1)
        motion_translation = curr2futu_ego[:, :, :3, 3].reshape(
            bsz, self.futu_number, -1)
        motion_rt = torch.cat([motion_rotation, motion_translation], dim=-1)
        return self.motion_embed(motion_rt)

    def _format_bev_layout_sequence(self, bev_layout):
        if bev_layout is None:
            return None
        if bev_layout.dim() != 5:
            raise ValueError(
                f'Expected bev_layout with 5 dims, got '
                f'{tuple(bev_layout.shape)}.')
        if bev_layout.size(2) == self.bev_layout_in_channels:
            layout = bev_layout.float()
        elif bev_layout.size(-1) == self.bev_layout_in_channels:
            layout = rearrange(bev_layout, 'b t h w c -> b t c h w').float()
        else:
            raise ValueError(
                'Cannot infer channel dimension for bev_layout with shape '
                f'{tuple(bev_layout.shape)} and '
                f'{self.bev_layout_in_channels} channels.')

        # Raw Layout rotated clockwise matches external OCC [X,Y].
        # The ray tokenizer stores its BEV feature map internally as [Y,X],
        # therefore rot90_cw(raw).T == raw.flip(axis0).
        return layout.flip(-2)

    def _select_future_bev_layout(self, bev_layout, current_offset=0):
        if not self.use_bev_layout:
            return None
        if bev_layout is None:
            raise ValueError(
                'BEV layout conditioning is enabled but bev_layout is missing '
                'from batch_inputs_dict.')

        bev_layout = self._format_bev_layout_sequence(bev_layout)
        total_frames = bev_layout.size(1)
        future_layout = []
        for step in range(1, self.futu_number + 1):
            frame_index = self.prev_number + current_offset + step
            if frame_index < total_frames:
                future_layout.append(bev_layout[:, frame_index])
            else:
                future_layout.append(bev_layout[:, -1])
        return torch.stack(future_layout, dim=1)

    def _future_layout_embedding(self, bev_layout, current_offset=0, dtype=None):
        future_layout = self._select_future_bev_layout(
            bev_layout, current_offset=current_offset)
        if future_layout is None:
            return None
        if dtype is not None:
            future_layout = future_layout.to(dtype=dtype)
        return self.bev_layout_embed(future_layout)

    def _rollout_ego_chunk(self, curr2futu_ego, current_offset):
        """Build ego-motion from rollout current frame to next model horizon."""
        available_future = curr2futu_ego.size(1)
        if available_future <= 0:
            raise ValueError(
                'Rollout requires at least one future ego transform.')
        if current_offset > available_future:
            raise ValueError(
                'Not enough curr2futu_ego transforms for rollout current '
                f'offset {current_offset}; got {available_future}.')
        if current_offset + 1 > available_future:
            raise ValueError(
                'Not enough curr2futu_ego transforms for the next rollout '
                f'frame at offset {current_offset}; got {available_future}.')

        bsz = curr2futu_ego.size(0)
        if current_offset == 0:
            base = torch.eye(
                4,
                device=curr2futu_ego.device,
                dtype=curr2futu_ego.dtype).unsqueeze(0).expand(bsz, -1, -1)
        else:
            base = curr2futu_ego[:, current_offset - 1]
        base_inv = torch.linalg.inv(base)

        chunk = []
        for step in range(1, self.futu_number + 1):
            future_index = current_offset + step
            if future_index <= available_future:
                future = curr2futu_ego[:, future_index - 1]
            else:
                future = curr2futu_ego[:, -1]
            chunk.append(future @ base_inv)
        return torch.stack(chunk, dim=1)

    def _flow_timestep_embed(self, timesteps, batch_size, device, dtype):
        timesteps = timesteps.to(device=device, dtype=dtype)
        timesteps = timesteps / self.flow_scheduler.config.num_train_timesteps
        return timesteps.view(batch_size, 1, 1, 1).expand(-1, self.futu_number, -1, -1)

    def _scale_flow_noise(self, sample, sigmas, noise):
        while sigmas.dim() < sample.dim():
            sigmas = sigmas.unsqueeze(-1)
        return sigmas * noise + (1.0 - sigmas) * sample

    def _sample_train_flow_timesteps(self, batch_size, device, dtype):
        scheduler_sigmas = self.flow_scheduler.sigmas.to(device=device, dtype=dtype)
        timestep_indices = torch.randint(
            0, len(scheduler_sigmas), (batch_size,), device=device)
        sigmas = scheduler_sigmas[timestep_indices].clamp(min=1e-5, max=1.0)
        timesteps = sigmas * float(self.flow_num_train_timesteps)
        return timesteps, sigmas

    def _flow_matching_loss(self,
                            pred_velocity,
                            target_velocity,
                            sample_frame_weights=None):
        loss = F.mse_loss(pred_velocity, target_velocity, reduction='none')
        if sample_frame_weights is not None:
            weights = sample_frame_weights.to(
                device=loss.device, dtype=loss.dtype)
            while weights.dim() < loss.dim():
                weights = weights.unsqueeze(-1)
            loss = loss * weights
        return loss.mean()

    def _linear_quadratic_sigmas(self, num_steps):
        linear_steps = self.flow_inference_lq_linear_steps
        if linear_steps is None:
            linear_steps = num_steps // 2
        linear_steps = int(linear_steps)
        if linear_steps <= 0 or linear_steps >= num_steps:
            raise ValueError(
                'flow_inference_lq_linear_steps must be in [1, '
                'flow_num_steps - 1].')

        threshold = self.flow_inference_lq_threshold
        linear_sigma_schedule = [
            i * threshold / linear_steps for i in range(linear_steps)
        ]
        threshold_step_diff = linear_steps - threshold * num_steps
        quadratic_steps = num_steps - linear_steps
        quadratic_coef = (
            threshold_step_diff / (linear_steps * quadratic_steps**2))
        linear_coef = (
            threshold / linear_steps -
            2 * threshold_step_diff / (quadratic_steps**2))
        const = quadratic_coef * (linear_steps**2)
        quadratic_sigma_schedule = [
            quadratic_coef * (i**2) + linear_coef * i + const
            for i in range(linear_steps, num_steps)
        ]
        sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule
        return [1.0 - sigma for sigma in sigma_schedule]

    def _set_inference_timesteps(self, scheduler, device):
        num_steps = max(int(self.flow_num_steps), 1)
        if self.flow_inference_schedule == 'default':
            scheduler.set_timesteps(
                num_inference_steps=num_steps,
                device=device)
        elif self.flow_inference_schedule == 'linear_quadratic':
            scheduler.set_timesteps(
                sigmas=self._linear_quadratic_sigmas(num_steps),
                device=device)
        else:
            raise ValueError(
                f'Unsupported flow_inference_schedule: '
                f'{self.flow_inference_schedule}')
        scheduler.set_begin_index(0)

    def _build_inference_scheduler(self):
        return FlowMatchEulerDiscreteScheduler.from_config(
            self.flow_scheduler.config)

    def encode_context(self, curr_bev_latent, memo_bev_latent):
        cond_latent = torch.cat([memo_bev_latent, curr_bev_latent], dim=1)
        bsz, context_frames = cond_latent.shape[:2]
        context = self._patchify_sequence(cond_latent)
        context = self.cond_proj(context)

        temporal_pos = self.context_temporal_pos_embed[:, :context_frames]
        temporal_pos = temporal_pos.unsqueeze(2).expand(-1, -1, self.seq_len, -1)
        spatial_pos = self.spatial_pos_embed.unsqueeze(1).expand(-1, context_frames, -1, -1)
        context = context + temporal_pos + spatial_pos

        reference_points, spatial_shapes, level_start_index = \
            self._build_reference_points(
                bsz, context_frames, context.device, context.dtype)

        for block in self.context_blocks:
            context = block(
                future=context,
                context=context,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index)

        return self.context_norm(context)

    def predict_flow_velocity(self, noisy_futu_tokens, context, curr2futu_ego,
                              flow_t, bev_layout_embed=None):
        bsz = noisy_futu_tokens.size(0)
        futu_x = self.noisy_proj(noisy_futu_tokens)

        temporal_pos = self.future_temporal_pos_embed.unsqueeze(2).expand(-1, -1, self.seq_len, -1)
        spatial_pos = self.spatial_pos_embed.unsqueeze(1).expand(-1, self.futu_number, -1, -1)
        futu_x = futu_x + temporal_pos + spatial_pos

        motion_embed = self._motion_embedding(curr2futu_ego).unsqueeze(2)
        time_embed = self.time_embed(flow_t.reshape(bsz, self.futu_number, 1))
        time_embed = time_embed.unsqueeze(2)
        futu_x = futu_x + motion_embed + time_embed
        if bev_layout_embed is not None:
            adapter_input = torch.cat([futu_x, bev_layout_embed], dim=-1)
            futu_x = futu_x + self.bev_layout_adapter(adapter_input)

        reference_points, spatial_shapes, level_start_index = \
            self._build_reference_points(
                bsz, self.futu_number, futu_x.device, futu_x.dtype)

        for block in self.flow_blocks:
            futu_x = block(
                future=futu_x,
                context=context,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index)

        futu_x = self.flow_norm(futu_x)
        velocity = self.velocity_proj(futu_x)
        return velocity

    def extract_feat(
            self, 
            curr_bev_latent, 
            memo_bev_latent, 
            curr2futu_ego,
            noisy_futu_tokens, 
            flow_t,
            bev_layout_embed=None,
        ):
        context = self.encode_context(curr_bev_latent, memo_bev_latent)
        return self.predict_flow_velocity(
            noisy_futu_tokens,
            context,
            curr2futu_ego,
            flow_t,
            bev_layout_embed=bev_layout_embed)

    def _future_latent_chunk(self, bev_latent, future_offset):
        start = self.prev_number + 1 + future_offset
        end = start + self.futu_number
        if bev_latent.size(1) < end:
            return None
        return bev_latent[:, start:end]

    def _compute_training_chunk_losses(self, curr_bev_latent,
                                       memo_bev_latent, curr2futu_ego,
                                       batch_inputs_dict,
                                       batch_data_samples=None,
                                       future_offset=0,
                                       loss_suffix='', loss_weight=1.0):
        bev_latent = batch_inputs_dict['bev_latent']
        futu_bev_latent = self._future_latent_chunk(
            bev_latent, future_offset)
        if futu_bev_latent is None:
            return {}

        bsz = bev_latent.size(0)
        target_tokens = self._patchify_sequence(futu_bev_latent)
        noise = torch.randn_like(target_tokens)
        timesteps, sigmas = self._sample_train_flow_timesteps(
            bsz, target_tokens.device, target_tokens.dtype)
        noisy_tokens = self._scale_flow_noise(target_tokens, sigmas, noise)
        flow_t = self._flow_timestep_embed(
            timesteps, bsz, target_tokens.device, target_tokens.dtype)
        target_velocity = noise - target_tokens
        ego_chunk = self._rollout_ego_chunk(curr2futu_ego, future_offset)
        bev_layout_embed = self._future_layout_embedding(
            batch_inputs_dict.get('bev_layout'),
            current_offset=future_offset,
            dtype=target_tokens.dtype)

        context = self.encode_context(curr_bev_latent, memo_bev_latent)
        pred_velocity = self.predict_flow_velocity(
            noisy_tokens,
            context,
            ego_chunk,
            flow_t,
            bev_layout_embed=bev_layout_embed)

        suffix = f'_{loss_suffix}' if loss_suffix else ''
        loss_dict = {
            f'loss_flow_matching{suffix}': (
                self._flow_matching_loss(
                    pred_velocity,
                    target_velocity,
                    sample_frame_weights=self._future_padding_frame_weights(
                        batch_data_samples,
                        future_offset,
                        pred_velocity.device,
                        pred_velocity.dtype)) *
                loss_weight)
        }
        loss_dict.update(self._range_lidar_aux_losses(
            pred_velocity,
            noisy_tokens,
            sigmas,
            batch_inputs_dict,
            batch_data_samples,
            future_offset=future_offset,
            loss_suffix=loss_suffix,
            loss_weight=loss_weight))
        return loss_dict

    def _select_training_chunk_indices(self):
        if self.train_autoregressive_chunk_mode == 'all':
            return list(range(self.train_autoregressive_chunks))

        chunk_idx = (
            self._train_autoregressive_chunk_cursor %
            self.train_autoregressive_chunks)
        self._train_autoregressive_chunk_cursor += 1
        return [chunk_idx]

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        bev_latent = batch_inputs_dict['bev_latent']
        curr2futu_ego = [
            torch.from_numpy(data_sample.metainfo['curr2futu_ego']).to(
                bev_latent.device).float()
            for data_sample in batch_data_samples
        ]
        curr2futu_ego = torch.stack(curr2futu_ego, dim=0)
        start_of_sequence = [
            data_sample.metainfo['start_of_sequence']
            for data_sample in batch_data_samples
        ]
        curr_bev_latent = bev_latent[:, self.prev_number:self.prev_number + 1]
        memo_bev_latent = self.get_memo_latent(
            curr_bev_latent, start_of_sequence)

        selected_chunks = self._select_training_chunk_indices()
        selected_chunk_set = set(selected_chunks)
        loss_dict = {}
        rollout_curr = curr_bev_latent
        rollout_memo = memo_bev_latent
        generated = 0
        for chunk_idx in range(max(selected_chunks) + 1):
            if chunk_idx > 0:
                with torch.no_grad():
                    ego_chunk = self._rollout_ego_chunk(
                        curr2futu_ego, generated)
                    pred_tokens = self.sample_future_tokens(
                        rollout_curr,
                        rollout_memo,
                        ego_chunk,
                        bev_layout=batch_inputs_dict.get('bev_layout'),
                        current_offset=generated)
                    pred_chunk = self._tokens_to_latent_sequence(
                        pred_tokens, bev_latent.size(0))
                    rollout_memo = self._append_memo_latent(
                        rollout_memo, pred_chunk)
                    rollout_curr = pred_chunk[:, -1:]
                    generated += self.futu_number

            if chunk_idx not in selected_chunk_set:
                continue

            use_stable_loss_names = (
                self.train_autoregressive_chunk_mode == 'round_robin')
            loss_suffix = (
                '' if chunk_idx == 0 or use_stable_loss_names
                else f'ar{chunk_idx}')
            loss_weight = (
                1.0 if chunk_idx == 0
                else self.train_autoregressive_loss_weight)
            chunk_losses = self._compute_training_chunk_losses(
                rollout_curr,
                rollout_memo,
                curr2futu_ego,
                batch_inputs_dict,
                batch_data_samples=batch_data_samples,
                future_offset=generated,
                loss_suffix=loss_suffix,
                loss_weight=loss_weight)
            if not chunk_losses:
                break
            loss_dict.update(chunk_losses)

        self.update_memo_latent(curr_bev_latent)
        return loss_dict

    @torch.no_grad()
    def sample_future_tokens(self, curr_bev_latent, memo_bev_latent,
                             curr2futu_ego, bev_layout=None,
                             current_offset=0):
        bsz = curr_bev_latent.size(0)
        device = curr_bev_latent.device
        dtype = curr_bev_latent.dtype
        context = self.encode_context(curr_bev_latent, memo_bev_latent)
        bev_layout_embed = self._future_layout_embedding(
            bev_layout,
            current_offset=current_offset,
            dtype=dtype)
        tokens = torch.randn(
            bsz, self.futu_number, self.seq_len, self.token_embed_dim,
            device=device, dtype=dtype
        )

        scheduler = self._build_inference_scheduler()
        self._set_inference_timesteps(scheduler, device)
        
        for timestep in scheduler.timesteps:
            timesteps = timestep.repeat(bsz)
            flow_t = self._flow_timestep_embed(timesteps, bsz, device, dtype)
            velocity = self.predict_flow_velocity(
                tokens,
                context,
                curr2futu_ego,
                flow_t,
                bev_layout_embed=bev_layout_embed)
            tokens = scheduler.step(velocity, timestep, tokens, return_dict=False)[0]

        return tokens

    def _tokens_to_latent_sequence(self, tokens, batch_size):
        tokens = rearrange(tokens, 'b t s c -> (b t) s c')
        latents = self.unpatchify(tokens)
        return rearrange(
            latents, '(b t) c h w -> b t c h w', b=batch_size)

    @torch.no_grad()
    def rollout_future_latents(self, curr_bev_latent, memo_bev_latent,
                               curr2futu_ego, future_number=None,
                               rollout_stride=None, bev_layout=None,
                               progress_callback=None):
        future_number = (
            self.rollout_future_number
            if future_number is None else int(future_number))
        if future_number is None:
            future_number = self.futu_number
        if future_number <= 0:
            raise ValueError('future_number must be positive.')
        rollout_stride = (
            self.rollout_stride
            if rollout_stride is None else int(rollout_stride))
        if not 1 <= rollout_stride <= self.futu_number:
            raise ValueError(
                'rollout_stride must be within [1, futu_number].')

        bsz = curr_bev_latent.size(0)
        rollout_curr = curr_bev_latent
        rollout_memo = memo_bev_latent.clone()
        pred_latents = []
        generated = 0

        while generated < future_number:
            ego_chunk = self._rollout_ego_chunk(curr2futu_ego, generated)
            pred_tokens = self.sample_future_tokens(
                rollout_curr,
                rollout_memo,
                ego_chunk,
                bev_layout=bev_layout,
                current_offset=generated)
            pred_chunk = self._tokens_to_latent_sequence(pred_tokens, bsz)

            accept_frames = min(
                rollout_stride, future_number - generated)
            accepted = pred_chunk[:, :accept_frames]
            pred_latents.append(accepted)

            rollout_memo = self._append_memo_latent(
                rollout_memo, accepted)
            rollout_curr = accepted[:, -1:]
            generated += accept_frames
            if progress_callback is not None:
                progress_callback(accept_frames)

        return torch.cat(pred_latents, dim=1)

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        bev_latent = batch_inputs_dict['bev_latent']
        bsz = bev_latent.size(0)
        curr2futu_ego = [
            torch.from_numpy(data_sample.metainfo['curr2futu_ego']).to(
                bev_latent.device).float()
            for data_sample in batch_data_samples
        ]
        curr2futu_ego = torch.stack(curr2futu_ego, dim=0)
        start_of_sequence = [
            data_sample.metainfo['start_of_sequence']
            for data_sample in batch_data_samples
        ]
        curr_bev_latent = bev_latent[:, self.prev_number:self.prev_number + 1]
        memo_bev_latent = self.get_memo_latent(curr_bev_latent, start_of_sequence)
        pred_futu_latent = self.rollout_future_latents(
            curr_bev_latent,
            memo_bev_latent,
            curr2futu_ego,
            bev_layout=batch_inputs_dict.get('bev_layout'))
        pred_future_number = pred_futu_latent.size(1)
        self.update_memo_latent(curr_bev_latent)

        pred_latent = torch.cat([curr_bev_latent, pred_futu_latent], dim=1)
        pred_latent = rearrange(pred_latent, 'b t c h w -> (b t) c h w')

        frames_per_sample = pred_future_number + 1
        pred_voxel_semantics, pred_points = self.obtain_data_from_latent(
            pred_latent,
            batch_data_samples=batch_data_samples,
            frames_per_sample=frames_per_sample)
        pred_voxel_semantics = rearrange(
            pred_voxel_semantics, '(b t) h w z -> b t h w z', b=bsz)

        targ_voxel_semantics = batch_inputs_dict['voxel_semantics'][
            :, self.prev_number:self.prev_number + frames_per_sample]

        if not self.preserve_full_pred_points:
            post_process_points = []
            for pts in pred_points:
                if pts.shape[0] > 50000:
                    post_process_points.append(
                        torch.ones(
                            (1, pts.shape[1]),
                            device=pts.device,
                            dtype=pts.dtype))
                else:
                    post_process_points.append(pts)
            pred_points = post_process_points

        targ_range_points = None
        if not self.use_projected_gt:
            targ_range_points = self._target_points_from_range(
                batch_inputs_dict, frames_per_sample)

        pred_data_samples = []
        for i in range(bsz):
            pred_data_sample = Det3DDataSample()
            pred_data_sample.set_metainfo(batch_data_samples[i].metainfo)
            pred_data_sample.pred_voxel_semantics = pred_voxel_semantics[i]
            pred_data_sample.targ_voxel_semantics = targ_voxel_semantics[i]
            pred_data_sample.pred_points = pred_points[
                i * frames_per_sample:(i + 1) * frames_per_sample]
            if self.use_projected_gt:
                frames = batch_data_samples[i].metainfo['fixed_range_imgs'][
                    self.prev_number:self.prev_number + frames_per_sample]
                if len(frames) != frames_per_sample or any(
                        tuple(frame.shape[:2]) != tuple(self.vae.range_resolution)
                        for frame in frames):
                    raise ValueError('Projected GT must match decoder resolution and horizon.')
                pred_data_sample.targ_points = [
                    projected_points(
                        frame, pred_data_sample.pred_points[0].device,
                        self.vae.fov_up_rad, self.vae.fov_down_rad,
                        self.vae.lidar_utils.min_depth, self.vae.lidar_utils.max_depth)
                    for frame in frames]
            else:
                pred_data_sample.targ_points = targ_range_points[i]
            pred_data_samples.append(pred_data_sample)
        return pred_data_samples

    def _forward(self, batch_inputs_dict, batch_data_samples=None, **kwargs):
        return self.predict(batch_inputs_dict, batch_data_samples, **kwargs)
