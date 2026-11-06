"""Unified OCC/LiDAR variational autoencoder.

The model fuses same-frame OCC and LiDAR voxel features before the shared
encoder. OCC is decoded through factorized occupancy/semantic heads, while a
ray-native categorical decoder reconstructs LiDAR depth and visibility.
"""

import math
import os

import mmengine
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmengine.model import BaseModel

from projects.losses.lovasz_softmax import lovasz_softmax
from projects.models.tokenizer.quantizer import DiagonalGaussianDistribution


class LiDARUtility(nn.Module):
    """Normalize metric range images and cache their LiDAR ray angles."""

    def __init__(self,
                 min_depth=1.45,
                 max_depth=80.0,
                 resolution=(32, 1024),
                 fov_up=10.0,
                 fov_down=-30.0):
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.resolution = resolution

        height, width = resolution
        elevation = 1 - torch.arange(height, dtype=torch.float32) / height
        elevation = elevation * (fov_up - fov_down) + fov_down
        azimuth = 1 - torch.arange(width, dtype=torch.float32) / width
        azimuth = azimuth * 360.0 - 180.0
        elevation, azimuth = torch.meshgrid(
            elevation, azimuth, indexing='ij')
        ray_angles = torch.stack([elevation, azimuth])[None].deg2rad()
        self.register_buffer('ray_angles', ray_angles.float())

    @staticmethod
    def normalize(value):
        return value * 2 - 1

    @staticmethod
    def denormalize(value):
        return (value + 1) / 2

    def convert_depth(self, metric_depth):
        valid = ((metric_depth > self.min_depth) &
                 (metric_depth < self.max_depth)).float()
        normalized = torch.log2(metric_depth + 1) / np.log2(
            self.max_depth + 1)
        return normalized.clamp(0, 1) * valid


@MODELS.register_module()
class OccLiDARVAE(BaseModel):
    """Encode and reconstruct aligned OCC and LiDAR observations.

    The latent and decoder trunk are shared. OCC logits factorize existence
    from conditional semantic identity. LiDAR logits are predicted directly
    in ray space with detached OCC probabilities as geometric context.
    """

    def __init__(self,
                 # Shared latent backbone.
                 shared_encoder,
                 shared_decoder,
                 occ_focal_loss,
                 data_preprocessor=None,
                 # Scene geometry.
                 pc_range=(-40.0, -40.0, -1.0, 40.0, 40.0, 5.4),
                 voxel_size=(0.4, 0.4, 0.4),
                 range_resolution=(32, 1024),
                 min_depth=1.45,
                 max_depth=80.0,
                 fov_up=10.0,
                 fov_down=-30.0,
                 range_sample_depth_bins=192,
                 ray_depth_sampling='log',
                 # OCC branch.
                 occ_num_classes=18,
                 free_class=17,
                 occ_class_embeds_dim=16,
                 occ_stem_channels=64,
                 free_logit_bias=2.0,
                 occupancy_prior_logit=0.833213344056216,
                 # LiDAR branch.
                 range_voxel_channels=2,
                 range_stem_channels=64,
                 lidar_volume_channels=32,
                 lidar_ray_channels=8,
                 occ_ray_condition_channels=4,
                 ray_hidden_channels=32,
                 occ_support_log_weight=0.25,
                 range_shared_grad_scale=0.25,
                 intensity_value=1.0,
                 # Shared representation.
                 scene_feature_channels=64,
                 latent_channels=16,
                 # Training objectives.
                 occ_loss_weight=1.0,
                 range_loss_weight=1.0,
                 kl_loss_weight=1e-6,
                 valid_pixel_loss_weight=1.0,
                 empty_pixel_loss_weight=0.1,
                 range_depth_loss_weight=2.0,
                 range_edge_loss_weight=2.0,
                 range_edge_threshold=0.08,
                 # Runtime output.
                 save_latent_root=None,
                 preserve_ray_geometry=False,
                 use_projected_gt=False):
        self.preserve_ray_geometry = bool(preserve_ray_geometry)
        self.use_projected_gt = bool(use_projected_gt)
        self.semantic_logit_gauge_bias = float(free_logit_bias)
        super().__init__(data_preprocessor=data_preprocessor)

        # Model dimensions and modality contracts.
        self.occ_num_classes = int(occ_num_classes)
        self.free_class = int(free_class)
        self.occ_class_embeds_dim = int(occ_class_embeds_dim)
        self.range_voxel_channels = int(range_voxel_channels)
        self.scene_feature_channels = int(scene_feature_channels)
        self.latent_channels = int(latent_channels)

        # Scene and ray geometry.
        self.range_resolution = tuple(range_resolution)
        self.range_sample_depth_bins = int(range_sample_depth_bins)
        self.ray_depth_sampling = ray_depth_sampling
        self.fov_up_rad = math.radians(float(fov_up))
        self.fov_down_rad = math.radians(float(fov_down))
        self.intensity_value = float(intensity_value)
        self._init_scene_geometry(
            pc_range=pc_range,
            voxel_size=voxel_size,
            min_depth=min_depth,
            max_depth=max_depth,
            fov_up=fov_up,
            fov_down=fov_down)
        self._ray_geometry_cache = {}

        # Objective weights.
        self.occ_loss_weight = float(occ_loss_weight)
        self.range_loss_weight = float(range_loss_weight)
        self.kl_loss_weight = float(kl_loss_weight)
        self.valid_pixel_loss_weight = float(valid_pixel_loss_weight)
        self.empty_pixel_loss_weight = float(empty_pixel_loss_weight)
        self.range_depth_loss_weight = float(range_depth_loss_weight)
        self.range_edge_loss_weight = float(range_edge_loss_weight)
        self.range_edge_threshold = float(range_edge_threshold)

        # Shared encoder and decoder. Keep module creation order stable because
        # scratch experiments depend on deterministic seeded initialization.
        self.quantizer = DiagonalGaussianDistribution
        self.occ_class_embeds = nn.Embedding(
            self.occ_num_classes, self.occ_class_embeds_dim)
        occ_in_channels = self.grid_z * self.occ_class_embeds_dim
        range_in_channels = self.grid_z * self.range_voxel_channels
        self.occ_stem = self._make_stem(occ_in_channels, int(occ_stem_channels))
        self.range_stem = self._make_stem(range_in_channels, int(range_stem_channels))
        self.shared_encoder = MODELS.build(shared_encoder)
        self.shared_decoder = MODELS.build(shared_decoder)

        # OCC objective and decoder branch.
        self.occ_focal_loss = (
            occ_focal_loss if isinstance(occ_focal_loss, nn.Module)
            else MODELS.build(occ_focal_loss))
        self._build_occ_decoder(occupancy_prior_logit)

        # LiDAR decoder branch.
        self.lidar_volume_channels = int(lidar_volume_channels)
        self.lidar_ray_channels = int(lidar_ray_channels)
        self.occ_ray_condition_channels = int(occ_ray_condition_channels)
        self.ray_hidden_channels = int(ray_hidden_channels)
        self.occ_support_log_weight = float(occ_support_log_weight)
        self.range_shared_grad_scale = float(range_shared_grad_scale)
        self._build_lidar_decoder()

        # Runtime output.
        self.save_latent_root = save_latent_root
        if self.save_latent_root is not None:
            mmengine.mkdir_or_exist(self.save_latent_root)

    def _apply(self, fn):
        self._ray_geometry_cache.clear()
        return super()._apply(fn)

    def _init_scene_geometry(self,
                             pc_range,
                             voxel_size,
                             min_depth,
                             max_depth,
                             fov_up,
                             fov_down):
        self.register_buffer(
            'pc_range', torch.tensor(pc_range, dtype=torch.float32),
            persistent=False)
        self.register_buffer(
            'voxel_size', torch.tensor(voxel_size, dtype=torch.float32),
            persistent=False)
        grid_size = ((self.pc_range[3:] - self.pc_range[:3]) /
                     self.voxel_size).long()
        self.grid_x, self.grid_y, self.grid_z = (
            int(size) for size in grid_size)

        self.lidar_utils = LiDARUtility(
            min_depth=min_depth,
            max_depth=max_depth,
            resolution=self.range_resolution,
            fov_up=fov_up,
            fov_down=fov_down)

    def _build_occ_decoder(self, occupancy_prior_logit):
        self.occupancy_head = nn.Conv2d(self.scene_feature_channels, self.grid_z, kernel_size=1)
        self.conditional_semantic_head = nn.Conv2d(
            self.scene_feature_channels,
            self.grid_z * (self.occ_num_classes - 1),
            kernel_size=1)
        self._init_factorized_prior(float(occupancy_prior_logit))

    def _build_lidar_decoder(self):
        self.lidar_volume_head = nn.Conv2d(
            self.scene_feature_channels,
            self.grid_z * self.lidar_volume_channels,
            kernel_size=1)
        self.lidar_ray_volume_proj = nn.Conv3d(
            self.lidar_volume_channels,
            self.lidar_ray_channels,
            kernel_size=1)
        self.occ_ray_condition_proj = nn.Conv3d(
            self.occ_num_classes,
            self.occ_ray_condition_channels,
            kernel_size=1)

        ray_input_channels = (self.lidar_ray_channels + self.occ_ray_condition_channels + 2)
        self.ray_depth_head = nn.Sequential(
            nn.Conv3d(
                ray_input_channels,
                self.ray_hidden_channels,
                kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv3d(self.ray_hidden_channels, 1, kernel_size=1))
        empty_input_channels = (
            self.lidar_ray_channels + self.occ_ray_condition_channels + 1)
        self.ray_empty_head = nn.Sequential(
            nn.Conv2d(
                empty_input_channels,
                self.ray_hidden_channels,
                kernel_size=3,
                padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.ray_hidden_channels, 1, kernel_size=1))

        nn.init.zeros_(self.ray_depth_head[-1].weight)
        nn.init.zeros_(self.ray_depth_head[-1].bias)
        nn.init.zeros_(self.ray_empty_head[-1].weight)
        nn.init.zeros_(self.ray_empty_head[-1].bias)

    @staticmethod
    def _make_stem(in_channels, out_channels):
        groups = min(32, out_channels)
        while out_channels % groups != 0:
            groups -= 1
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True))

    def _init_factorized_prior(self, occupancy_prior_logit):
        nn.init.normal_(self.occupancy_head.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.occupancy_head.bias, occupancy_prior_logit)
        nn.init.normal_(self.conditional_semantic_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.conditional_semantic_head.bias)

    @staticmethod
    def _as_point_tensor(points):
        return points.tensor if hasattr(points, 'tensor') else points

    @staticmethod
    def _meta_value(data_sample, key, default=None):
        if data_sample is None:
            return default
        if hasattr(data_sample, 'metainfo') and key in data_sample.metainfo:
            return data_sample.metainfo[key]
        if hasattr(data_sample, key):
            return getattr(data_sample, key)
        return default

    def _sample_matrix(self, data_sample, key, device, dtype):
        value = self._meta_value(data_sample, key)
        if value is None:
            return torch.eye(4, device=device, dtype=dtype)
        return torch.as_tensor(value, device=device, dtype=dtype)

    def _ego2lidar(self, data_sample, device, dtype):
        return self._sample_matrix(data_sample, 'ego2lidar', device, dtype)

    def _lidar2ego(self, data_sample, device, dtype):
        value = self._meta_value(data_sample, 'lidar2ego')
        if value is not None:
            return torch.as_tensor(value, device=device, dtype=dtype)
        return torch.linalg.inv(
            self._ego2lidar(data_sample, device, dtype))

    def _format_range_batch(self, range_imgs):
        if range_imgs.dim() == 5:
            range_imgs = rearrange(range_imgs, 'b f h w c -> (b f) c h w')
        elif range_imgs.dim() == 4 and range_imgs.shape[-1] in (1, 2):
            range_imgs = rearrange(range_imgs, 'b h w c -> b c h w')
        elif range_imgs.dim() == 3:
            range_imgs = range_imgs.unsqueeze(1)
        if range_imgs.dim() != 4:
            raise ValueError(
                f'Unsupported range input shape {tuple(range_imgs.shape)}.')
        return range_imgs[:, :1].float()

    def _expand_data_samples(self, data_samples, batch_size):
        if data_samples is None or len(data_samples) == 0:
            return [None] * batch_size
        frames_per_sample = max(batch_size // len(data_samples), 1)
        return [
            data_samples[min(index // frames_per_sample,
                             len(data_samples) - 1)]
            for index in range(batch_size)]

    def _range_rays_lidar(self, device, dtype):
        key = ('rays', device.type, device.index, dtype)
        cached = self._ray_geometry_cache.get(key)
        if cached is not None:
            return cached

        height, width = self.range_resolution
        v = torch.arange(height, device=device, dtype=dtype)
        u = torch.arange(width, device=device, dtype=dtype)
        v, u = torch.meshgrid(v, u, indexing='ij')
        fov_up = torch.as_tensor(
            self.fov_up_rad, device=device, dtype=dtype)
        fov_down = torch.as_tensor(
            self.fov_down_rad, device=device, dtype=dtype)
        fov_rad = fov_up - fov_down
        yaw = (1.0 - (u + 0.5) / width * 2.0) * math.pi
        pitch = (1.0 - (v + 0.5) / height) * fov_rad + fov_down
        rays = torch.stack([
            torch.cos(pitch) * torch.cos(yaw),
            torch.cos(pitch) * torch.sin(yaw),
            torch.sin(pitch)], dim=-1)
        self._ray_geometry_cache[key] = rays
        return rays

    def _points_to_range_batch(self, points_list, data_samples):
        height, width = self.range_resolution
        range_imgs = []
        for index, points in enumerate(points_list):
            points = self._as_point_tensor(points)
            xyz = points[:, :3]
            ego2lidar = self._ego2lidar(
                data_samples[index] if data_samples is not None else None,
                xyz.device, xyz.dtype)
            hom = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=1)
            xyz = (ego2lidar @ hom.T).T[:, :3]
            depth = torch.linalg.norm(xyz, dim=1)
            valid = ((depth >= self.lidar_utils.min_depth) &
                     (depth <= self.lidar_utils.max_depth))
            xyz, depth = xyz[valid], depth[valid]
            range_img = depth.new_zeros((height, width))
            if depth.numel():
                yaw = torch.atan2(xyz[:, 1], xyz[:, 0])
                pitch = torch.asin((xyz[:, 2] / depth).clamp(-1.0, 1.0))
                fov_up = depth.new_tensor(self.fov_up_rad)
                fov_down = depth.new_tensor(self.fov_down_rad)
                proj_x = torch.floor(
                    0.5 * (1.0 - yaw / math.pi) * width).long()
                proj_y = torch.floor(
                    (1.0 - (pitch - fov_down) / (fov_up - fov_down)) *
                    height).long()
                proj_x = proj_x.clamp(0, width - 1)
                proj_y = proj_y.clamp(0, height - 1)
                order = torch.argsort(depth, descending=True)
                range_img[proj_y[order], proj_x[order]] = depth[order]
            range_imgs.append(range_img.unsqueeze(0))
        return torch.stack(range_imgs)

    def _get_range_input(self, batch_inputs_dict, data_samples):
        if 'range_imgs' in batch_inputs_dict:
            return self._format_range_batch(batch_inputs_dict['range_imgs'])
        if 'points' not in batch_inputs_dict:
            raise KeyError('Expected "range_imgs" or "points" in inputs.')
        return self._points_to_range_batch(
            batch_inputs_dict['points'], data_samples)

    def _range_to_ego_xyz(self, range_imgs, data_samples):
        depth = range_imgs[:, 0]
        valid = ((depth > self.lidar_utils.min_depth) &
                 (depth < self.lidar_utils.max_depth))
        rays = self._range_rays_lidar(depth.device, depth.dtype)
        xyz_lidar = rays.unsqueeze(0) * depth.unsqueeze(-1)
        expanded_samples = self._expand_data_samples(
            data_samples, range_imgs.size(0))
        xyz_ego = []
        for batch_index, data_sample in enumerate(expanded_samples):
            lidar2ego = self._lidar2ego(
                data_sample, depth.device, depth.dtype)
            xyz = xyz_lidar[batch_index].reshape(-1, 3)
            hom = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=1)
            xyz = (lidar2ego @ hom.T).T[:, :3]
            xyz_ego.append(xyz.reshape(*depth.shape[1:], 3))
        return torch.stack(xyz_ego), valid

    def _range_to_voxel_features(self, range_imgs, data_samples):
        batch, _, height, width = range_imgs.shape
        xyz_ego, valid = self._range_to_ego_xyz(range_imgs, data_samples)
        norm_depth = self.lidar_utils.convert_depth(range_imgs[:, 0:1])
        pixel_features = [norm_depth[:, 0], valid.to(range_imgs.dtype)]
        while len(pixel_features) < self.range_voxel_channels:
            pixel_features.append(torch.zeros_like(pixel_features[0]))
        pixel_features = torch.stack(
            pixel_features[:self.range_voxel_channels], dim=1)
        relative = ((xyz_ego - self.pc_range[:3].view(1, 1, 1, 3)) /
                    self.voxel_size.view(1, 1, 1, 3))
        indices = torch.floor(relative).long()
        ix, iy, iz = indices[..., 0], indices[..., 1], indices[..., 2]
        inside = (valid & (ix >= 0) & (ix < self.grid_x) &
                  (iy >= 0) & (iy < self.grid_y) &
                  (iz >= 0) & (iz < self.grid_z))
        volumes = []
        volume_size = self.grid_z * self.grid_y * self.grid_x
        for batch_index in range(batch):
            volume = range_imgs.new_zeros(
                self.range_voxel_channels, volume_size)
            counts = range_imgs.new_zeros(1, volume_size)
            mask = inside[batch_index].reshape(-1)
            if mask.any():
                linear = (
                    iz[batch_index].reshape(-1)[mask] * self.grid_y *
                    self.grid_x + iy[batch_index].reshape(-1)[mask] *
                    self.grid_x + ix[batch_index].reshape(-1)[mask])
                features = pixel_features[batch_index].reshape(
                    self.range_voxel_channels, height * width)[:, mask]
                volume.index_add_(1, linear, features)
                counts.index_add_(
                    1, linear,
                    torch.ones(1, linear.numel(), device=linear.device,
                               dtype=volume.dtype))
                volume = volume / counts.clamp(min=1.0)
            volume = volume.reshape(
                self.range_voxel_channels, self.grid_z,
                self.grid_y, self.grid_x)
            volumes.append(volume.reshape(
                self.range_voxel_channels * self.grid_z,
                self.grid_y, self.grid_x))
        return torch.stack(volumes)

    def _occ_to_bev_features(self, voxel_semantics):
        """Convert dataset OCC ``[B, X, Y, Z]`` to internal ``[B, C, Y, X]``."""
        voxel_semantics = voxel_semantics.long()
        embeds = self.occ_class_embeds(voxel_semantics)
        return rearrange(embeds, 'b x y z c -> b (z c) y x')

    def encode_fused(self, batch_inputs_dict, data_samples=None):
        range_imgs = self._get_range_input(batch_inputs_dict, data_samples)
        occ = batch_inputs_dict['voxel_semantics']
      
        occ_features = self.occ_stem(self._occ_to_bev_features(occ))
        range_voxels = self._range_to_voxel_features(range_imgs, data_samples)
        range_features = self.range_stem(range_voxels)

        fused_features = torch.cat([occ_features, range_features], dim=1)
        posterior_params, shapes = self.shared_encoder(fused_features)
        posterior = self.quantizer(posterior_params)
        return posterior, shapes, range_imgs

    def _default_skip_shapes(self, latent):
        height, width = latent.shape[-2:]
        shapes = []
        for level in reversed(range(self.shared_decoder.num_resolutions)):
            if level not in self.shared_decoder.upsample_level:
                continue
            upsample = getattr(self.shared_decoder.up[level], 'upsample', None)
            stride = getattr(upsample, 'stride', 1)
            if isinstance(stride, tuple):
                height *= stride[0]
                width *= stride[1]
            else:
                height *= stride
                width *= stride
            shapes.append((int(height), int(width)))
        return list(reversed(shapes))

    def _decode_scene_state(self, latent, shapes=None):
        if shapes is None:
            shapes = self._default_skip_shapes(latent)
        scene_features = self.shared_decoder(latent, list(shapes))

        occupancy = rearrange(
            self.occupancy_head(scene_features),
            'b z y x -> b x y z')
        conditional = rearrange(
            self.conditional_semantic_head(scene_features),
            'b (z c) y x -> b x y z c',
            z=self.grid_z, c=self.occ_num_classes - 1)
        return scene_features, occupancy, conditional

    def decode_scene_components(self, latent, shapes=None):
        _, occupancy, conditional = self._decode_scene_state(latent, shapes)
        return occupancy, conditional

    def factorized_semantic_logits(self, occupancy, conditional):
        """Convert factorized occupancy and class scores to raw OCC logits."""
        conditional_log_prob = F.log_softmax(conditional.float(), dim=-1)
        gauge = (conditional.float().mean(dim=-1) +
                 self.semantic_logit_gauge_bias)
        free_logit = gauge
        nonfree_logits = (
            gauge.unsqueeze(-1) + occupancy.unsqueeze(-1) +
            conditional_log_prob)
        classes = []
        conditional_index = 0
        for class_index in range(self.occ_num_classes):
            if class_index == self.free_class:
                classes.append(free_logit)
            else:
                classes.append(nonfree_logits[..., conditional_index])
                conditional_index += 1
        return torch.stack(classes, dim=-1)

    def decode_scene_logits(self, latent, shapes=None):
        _, occupancy, conditional = self._decode_scene_state(latent, shapes)
        return self.factorized_semantic_logits(occupancy, conditional)

    def semantic_occupancy_logits(self, semantic_logits):
        """Recover the factorized non-free/free log odds."""
        nonfree_indices = [
            index for index in range(self.occ_num_classes)
            if index != self.free_class]
        nonfree = torch.logsumexp(
            semantic_logits[..., nonfree_indices].float(), dim=-1)
        free = semantic_logits[..., self.free_class].float()
        return nonfree - free

    def _ray_depth_samples(self, device, dtype):
        """
        Compute a set of depth samples and their normalized positions along
        LiDAR rays.  The positions are normalized to [-1, 1] for use in
        ``F.grid_sample`` with ``align_corners=False``.
        """
        key = ('depths', device.type, device.index, dtype)
        cached = self._ray_geometry_cache.get(key)
        if cached is not None:
            return cached

        if self.ray_depth_sampling == 'linear':
            depths = torch.linspace(
                self.lidar_utils.min_depth, self.lidar_utils.max_depth,
                self.range_sample_depth_bins, device=device, dtype=dtype)
            log_max = math.log2(self.lidar_utils.max_depth + 1.0)
            depth_pos = torch.log2(depths + 1.0) / log_max * 2.0 - 1.0
            result = (depths, depth_pos)
            self._ray_geometry_cache[key] = result
            return result

        log_max = math.log2(self.lidar_utils.max_depth + 1.0)
        min_pos = math.log2(self.lidar_utils.min_depth + 1.0) / log_max

        positions = torch.linspace(
            min_pos, 1.0, self.range_sample_depth_bins,
            device=device, dtype=dtype
        )
        depths = torch.exp2(positions * log_max) - 1.0
        result = (depths, positions * 2.0 - 1.0)
        self._ray_geometry_cache[key] = result
        return result

    def _ray_xyz_lidar(self, device, dtype):
        key = ('xyz', device.type, device.index, dtype)
        cached = self._ray_geometry_cache.get(key)
        if cached is not None:
            return cached

        height, width = self.range_resolution
        rays = self._range_rays_lidar(device, dtype)
        depths, _ = self._ray_depth_samples(device, dtype)
        xyz = depths.view(-1, 1, 1, 1) * rays.view(
            1, height, width, 3)
        self._ray_geometry_cache[key] = xyz
        return xyz

    def _ray_sample_grid(self, batch_size, data_samples, device, dtype):
        """
            Compute a sampling grid for LiDAR ray features in ego coordinates.
        """
        height, width = self.range_resolution
        xyz_lidar = self._ray_xyz_lidar(device, dtype)
        expanded_samples = self._expand_data_samples(data_samples, batch_size)

        grids, inside_masks = [], []
        for data_sample in expanded_samples:
            lidar2ego = self._lidar2ego(data_sample, device, dtype)
            xyz = xyz_lidar.reshape(-1, 3)
            hom = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=1)
            xyz = (lidar2ego @ hom.T).T[:, :3]
            grid = self._ego_xyz_to_grid(xyz)
            inside = ((grid >= -1.0) & (grid <= 1.0)).all(dim=-1)
            grids.append(grid.reshape(self.range_sample_depth_bins, height, width, 3))
            inside_masks.append(inside.reshape(self.range_sample_depth_bins, height, width))

        return torch.stack(grids), torch.stack(inside_masks)

    def _ego_xyz_to_grid(self, xyz):
        """Map metric ego xyz to an ``align_corners=False`` sampling grid.

        With this convention a voxel center maps exactly to its tensor index;
        using the same normalization with ``align_corners=True`` introduces an
        approximately half-voxel shift on every axis.
        """
        return ((xyz - self.pc_range[:3]) /
                (self.pc_range[3:] - self.pc_range[:3]) * 2.0 - 1.0)

    @staticmethod
    def _scale_gradient(value, scale):
        return value.detach() + (value - value.detach()) * scale

    def _occ_probability_volumes(self, occupancy, conditional):
        """
            Convert factorized occupancy and conditional semantic logits to
            semantic and occupied probability volumes.
        """
        occupied = occupancy.detach().float().sigmoid()
        conditional_probability = F.softmax(conditional.detach().float(), dim=-1)
        nonfree_probability = (occupied.unsqueeze(-1) * conditional_probability)
        class_probabilities = []
        conditional_index = 0
        for class_index in range(self.occ_num_classes):
            if class_index == self.free_class:
                class_probabilities.append(1.0 - occupied)
            else:
                class_probabilities.append(
                    nonfree_probability[..., conditional_index])
                conditional_index += 1
        semantic = torch.stack(class_probabilities, dim=-1)
        semantic = semantic.permute(0, 4, 3, 2, 1).contiguous()
        occupied = occupied.permute(0, 3, 2, 1).unsqueeze(1).contiguous()
        return semantic, occupied

    def decode_lidar_logits(
        self,
        scene_features,
        occupancy,
        conditional,
        data_samples=None
    ):
        shared_features = self._scale_gradient(scene_features, self.range_shared_grad_scale)
        lidar_volume = self.lidar_volume_head(shared_features)
        lidar_volume = rearrange(
            lidar_volume,
            'b (z c) y x -> b c z y x',
            z=self.grid_z, c=self.lidar_volume_channels)
        lidar_volume = self.lidar_ray_volume_proj(lidar_volume)

        semantic_volume, occupied_volume = self._occ_probability_volumes(occupancy, conditional)
        semantic_volume = self.occ_ray_condition_proj(semantic_volume)

        grid, inside = self._ray_sample_grid(
            lidar_volume.size(0), data_samples,
            lidar_volume.device, lidar_volume.dtype)
        lidar_samples = F.grid_sample(
            lidar_volume, grid, mode='bilinear', padding_mode='zeros',
            align_corners=False
        )
        semantic_samples = F.grid_sample(
            semantic_volume.to(lidar_volume.dtype),
            grid, mode='bilinear', padding_mode='zeros',
            align_corners=False
        )
        occupied_samples = F.grid_sample(
            occupied_volume.to(lidar_volume.dtype),
            grid, mode='bilinear', padding_mode='zeros',
            align_corners=False
        )

        _, depth_position = self._ray_depth_samples(lidar_volume.device, lidar_volume.dtype)
        depth_position = depth_position.view(1, 1, -1, 1, 1).expand(
            lidar_samples.size(0), 1, -1,
            lidar_samples.size(3), lidar_samples.size(4))
        inside_feature = inside.unsqueeze(1).to(lidar_volume.dtype)
        ray_features = torch.cat([
            lidar_samples, semantic_samples,
            depth_position, inside_feature,
        ], dim=1)
        depth_logits = self.ray_depth_head(ray_features).squeeze(1)

        support_prior = torch.log(occupied_samples.squeeze(1).clamp(min=1e-4))
        depth_logits = (depth_logits + self.occ_support_log_weight * support_prior)
        depth_logits = depth_logits.masked_fill(~inside, -20.0)

        empty_features = torch.cat([
            lidar_samples, semantic_samples, inside_feature,
        ], dim=1)
        denominator = inside_feature.sum(dim=2).clamp(min=1.0)
        empty_features = (empty_features * inside_feature).sum(dim=2) / denominator
        empty_logits = self.ray_empty_head(empty_features)

        return torch.cat([depth_logits, empty_logits], dim=1)

    def decode_joint_with_logits(
        self,
        latent,
        occ_shapes=None,
        data_samples=None,
        range_shapes=None
    ):
        del range_shapes
        # Joint decoding
        scene_features, occupancy, conditional = self._decode_scene_state(latent, occ_shapes)
        # Occ head
        semantic_logits = self.factorized_semantic_logits(occupancy, conditional)
        # LiDAR head
        range_logits = self.decode_lidar_logits(scene_features, occupancy, conditional, data_samples)
        return semantic_logits, range_logits

    def _prepare_range_target(self, range_imgs):
        """
            Bin the raw LiDAR depth values into discrete depth bins for cross-entropy
        """
        raw_depth = range_imgs[:, 0]
        valid = ((raw_depth > self.lidar_utils.min_depth) &
                 (raw_depth < self.lidar_utils.max_depth))
        depths, _ = self._ray_depth_samples(raw_depth.device, raw_depth.dtype)
        edges = (depths[:-1] + depths[1:]) * 0.5
        target_bins = torch.bucketize(raw_depth, edges).clamp(min=0, max=self.range_sample_depth_bins - 1)
        empty_index = self.range_sample_depth_bins
        target_bins = torch.where(
            valid, target_bins,
            target_bins.new_full(target_bins.shape, empty_index))
        return target_bins.long(), valid

    def _range_logits_to_depth(self, range_logits):
        indices = range_logits.argmax(dim=1)
        hit = indices < self.range_sample_depth_bins
        depths, _ = self._ray_depth_samples(range_logits.device, range_logits.dtype)
        indices = indices.clamp(max=self.range_sample_depth_bins - 1)
        return depths[indices] * hit.to(depths.dtype)

    def _range_logits_to_soft_normalized_depth(self, range_logits):
        hit_logits = range_logits[:, :self.range_sample_depth_bins].float()
        probabilities = hit_logits.softmax(dim=1)
        _, positions = self._ray_depth_samples(
            range_logits.device, probabilities.dtype)
        return (probabilities * positions.view(1, -1, 1, 1)).sum(dim=1)

    def _range_edge_weights(self, range_imgs, valid):
        if self.range_edge_loss_weight <= 0:
            return range_imgs.new_ones(valid.shape)
        depth = self.lidar_utils.normalize(self.lidar_utils.convert_depth(range_imgs[:, 0])).detach()
        edge = torch.zeros_like(valid)
        depth_right = torch.roll(depth, shifts=-1, dims=2)
        valid_right = torch.roll(valid, shifts=-1, dims=2)
        edge_right = ((depth - depth_right).abs() >
                      self.range_edge_threshold) & valid & valid_right
        edge |= edge_right | torch.roll(edge_right, shifts=1, dims=2)
        depth_down = depth[:, 1:] - depth[:, :-1]
        valid_down = valid[:, 1:] & valid[:, :-1]
        edge_down = ((depth_down.abs() > self.range_edge_threshold) &
                     valid_down)
        edge[:, 1:] |= edge_down
        edge[:, :-1] |= edge_down
        return 1.0 + edge.to(range_imgs.dtype) * self.range_edge_loss_weight

    def _compute_occ_loss(self, semantic_logits, targets):
        """
            Focal + Lovasz loss for occupancy reconstruction. 
        """
        logits = rearrange(semantic_logits, 'b x y z c -> b c x y z')
        targets = targets.long()

        focal = self.occ_focal_loss(logits, targets, None, ignore_index=255)
        lovasz = lovasz_softmax(torch.softmax(logits, dim=1), targets, ignore=255)
        return (focal + lovasz) * self.occ_loss_weight

    def _compute_range_loss(self, range_logits, range_imgs):
        """
            CE + Smooth L1 loss for LiDAR depth reconstruction.  
            The CE loss is weighted by the edge map and valid/empty pixel weights.  
            The Smooth L1 loss is only computed for valid pixels and weighted by the edge map.
        """
        target_bins, valid = self._prepare_range_target(range_imgs)
        cross_entropy = F.cross_entropy(
            range_logits.float(), target_bins, reduction='none')
        weights = torch.where(
            valid,
            cross_entropy.new_full(
                cross_entropy.shape, self.valid_pixel_loss_weight),
            cross_entropy.new_full(
                cross_entropy.shape, self.empty_pixel_loss_weight))
        edge_weights = self._range_edge_weights(range_imgs, valid)
        if self.range_edge_loss_weight > 0 and valid.any():
            edge_weights = torch.where(
                valid,
                edge_weights / edge_weights[valid].mean().clamp(min=1e-6),
                torch.ones_like(edge_weights))
        weights = weights * edge_weights
        losses = dict(loss_range_ce=(cross_entropy * weights).mean() * self.range_loss_weight)

        # Depth regression loss
        if self.range_depth_loss_weight > 0:
            target = self.lidar_utils.normalize(self.lidar_utils.convert_depth(range_imgs[:, 0]))
            prediction = self._range_logits_to_soft_normalized_depth(range_logits)

            if valid.any():
                regression = F.smooth_l1_loss(prediction, target, reduction='none')
                depth_loss = (regression[valid] * edge_weights[valid]).mean()
            else:
                depth_loss = prediction.sum() * 0.0
            losses['loss_range_depth'] = (depth_loss * self.range_depth_loss_weight * self.range_loss_weight)

        return losses

    def _save_latents(self, posterior, latent, data_samples):
        if self.save_latent_root is None or self.training or not data_samples:
            return
        frames_per_sample = max(latent.size(0) // len(data_samples), 1)
        for sample_index, data_sample in enumerate(data_samples):
            latent_index = min(sample_index * frames_per_sample, latent.size(0) - 1)
            token = self._meta_value(data_sample, 'token')
            if token is None:
                raise KeyError('A token is required when saving latents.')
            np.savez(
                os.path.join(self.save_latent_root, f'{token}.npz'),
                latent=latent[latent_index].detach().cpu().numpy(),
                posterior_mean=posterior.mean[latent_index].detach().cpu().numpy()
            )

    def extract_feat(self, batch_inputs_dict, data_samples=None):
        """
            Extract features from inputs and data samples.
            Training and Inference are both supported.  
        """
        posterior, shapes, range_imgs = self.encode_fused(batch_inputs_dict, data_samples)
        latent = posterior.sample()
        self._save_latents(posterior, latent, data_samples)
        semantic_logits, range_logits = self.decode_joint_with_logits(latent, shapes, data_samples)
        return dict(
            posterior=posterior,
            latent=latent,
            shapes=shapes,
            range_imgs=range_imgs,
            semantic_logits=semantic_logits,
            range_logits=range_logits)

    def loss(self, batch_inputs_dict, batch_data_samples=None, **kwargs):

        # --------------- Forward ---------------
        result = self.extract_feat(batch_inputs_dict, batch_data_samples)

        # --------------- Losses ---------------
        # Occupancy reconstruction loss
        loss_occ_recon = self._compute_occ_loss(result['semantic_logits'], batch_inputs_dict['voxel_semantics'])
        # Range reconstruction loss
        loss_range_dict = self._compute_range_loss(result['range_logits'], result['range_imgs'])
        # KL loss
        loss_kl = result['posterior'].kl().mean() * self.kl_loss_weight

        losses = dict(
            loss_occ_recon=loss_occ_recon,
            **loss_range_dict,
            loss_kl=loss_kl)

        return losses

    def _range_depth_to_points_lidar(self, depth):
        """
            Convert a LiDAR depth image to a point cloud in LiDAR coordinates.
        """
        mask = ((depth > self.lidar_utils.min_depth) &
                (depth < self.lidar_utils.max_depth))
      
        rays = self._range_rays_lidar(depth.device, depth.dtype)
        xyz = rays[mask] * depth[mask].unsqueeze(-1)
        # In this version, we do not include intensity values in the point cloud.
        intensity = torch.full((xyz.size(0), 1), self.intensity_value, device=depth.device, dtype=depth.dtype)

        return torch.cat([xyz, intensity], dim=1)

    def decode_joint(self,
                     latent,
                     occ_shapes=None,
                     data_samples=None,
                     range_shapes=None):
        semantic_logits, range_logits = self.decode_joint_with_logits(latent, occ_shapes, data_samples, range_shapes)
        return semantic_logits, self._range_logits_to_depth(range_logits)

    def decode_to_points(self, latent, data_samples=None, occ_shapes=None):
        _, depths = self.decode_joint(latent, occ_shapes, data_samples)
        return self.depths_to_points(depths)

    def depths_to_points(self, depths):
        return [
            self._range_depth_to_points_lidar(depth)
            for depth in depths
        ]

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Ray angles describe configured geometry, not learned weights.
        if self.preserve_ray_geometry:
            key = prefix + 'lidar_utils.ray_angles'
            if key in state_dict:
                state_dict[key] = self.lidar_utils.ray_angles.detach().clone()
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def predict(self, batch_inputs_dict, batch_data_samples=None, **kwargs):
        # ----------------- Forward -----------------
        result = self.extract_feat(batch_inputs_dict, batch_data_samples)

        # ----------------- Postprocess -----------------
        predicted_occ = result['semantic_logits'].argmax(dim=-1)
        predicted_depth = self._range_logits_to_depth(result['range_logits'])

        outputs = []
        sample_count = max(len(batch_data_samples), 1)
        frames_per_sample = max(predicted_depth.size(0) // sample_count, 1)
        for sample_index, data_sample in enumerate(batch_data_samples):
            frame_index = sample_index * frames_per_sample
            output = Det3DDataSample()
            output.pred_voxel_semantics = predicted_occ[frame_index]
            occ3d_target = data_sample.metainfo.get('gt_voxel_semantics')
            if occ3d_target is None:
                occ3d_target = batch_inputs_dict['voxel_semantics'][sample_index]
            else:
                occ3d_target = torch.as_tensor(
                    occ3d_target,
                    device=predicted_occ.device,
                    dtype=torch.long)
            output.targ_voxel_semantics = occ3d_target
            output.pred_points = self._range_depth_to_points_lidar(predicted_depth[frame_index])
            if self.use_projected_gt:
                output.targ_points = projected_points(
                    data_sample.metainfo['fixed_range_imgs'],
                    output.pred_points.device, self.fov_up_rad, self.fov_down_rad,
                    self.lidar_utils.min_depth, self.lidar_utils.max_depth)
            else:
                output.targ_points = self._range_depth_to_points_lidar(
                    result['range_imgs'][frame_index, 0])
            output.set_metainfo({
                key: value
                for key, value in data_sample.metainfo.items()
                if key != 'gt_voxel_semantics'
            })
            outputs.append(output)
        return outputs

    def _forward(self, batch_inputs_dict, batch_data_samples=None, **kwargs):
        return self.predict(
            batch_inputs_dict, batch_data_samples, **kwargs)

    def forward(self, inputs, data_samples=None, mode='tensor', **kwargs):
        if mode == 'loss':
            return self.loss(inputs, data_samples, **kwargs)
        if mode == 'predict':
            return self.predict(inputs, data_samples, **kwargs)
        if mode == 'tensor':
            return self._forward(inputs, data_samples, **kwargs)
        raise RuntimeError(f'Invalid mode {mode!r}.')


def projected_points(range_image, device, fov_up, fov_down, min_depth, max_depth):
    depth = torch.as_tensor(range_image, device=device, dtype=torch.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    assert depth.ndim == 2 and depth.shape[0] == 64
    height, width = depth.shape
    v, u = torch.meshgrid(torch.arange(height, device=device),
                          torch.arange(width, device=device), indexing='ij')
    pitch = (1-(v+.5)/height)*(fov_up-fov_down)+fov_down
    yaw = (1-(u+.5)/width*2)*math.pi
    rays = torch.stack((pitch.cos()*yaw.cos(),pitch.cos()*yaw.sin(),pitch.sin()),-1)
    valid = (depth > min_depth) & (depth < max_depth)
    return rays[valid]*depth[valid].unsqueeze(-1)
