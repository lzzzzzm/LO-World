# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss

from mmdet3d.registry import MODELS


def py_sigmoid_focal_loss(pred, target, weight=None, gamma=2.0, alpha=0.25):
    """Compute sigmoid focal loss from logits and one-hot targets in PyTorch."""
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                weight = weight.view(-1, 1)
            else:
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
        loss = loss * weight
    return loss.sum(-1).mean()


def sigmoid_focal_loss(pred, target, weight=None, gamma=2.0, alpha=0.25):
    """Compute CUDA sigmoid focal loss from logits and class-index targets."""
    loss = _sigmoid_focal_loss(pred.contiguous(), target.contiguous(), gamma,
                               alpha, None, 'none')
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                weight = weight.view(-1, 1)
            else:
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
        loss = loss * weight
    return loss.sum(-1).mean()


@MODELS.register_module()
class CustomFocalLoss(nn.Module):
    """Voxel focal loss with radial BEV weights and ignored-label masking.

    Class losses are summed per voxel, then averaged over valid voxels.
    """

    def __init__(self, gamma=2.0, alpha=0.25, loss_weight=100.0,
                 bev_h=200, bev_w=200):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.loss_weight = loss_weight

        xy, yx = torch.meshgrid([torch.arange(bev_h) - bev_h / 2,
                                torch.arange(bev_w) - bev_w / 2])
        c = torch.stack([xy, yx], 2)
        c = torch.norm(c, 2, -1)
        self.c = (c / c.max() + 1).cuda()

    def forward(self, pred, target, weight=None, ignore_index=255):
        """Compute loss for logits [B,C,H,W,D] and labels [B,H,W,D]."""
        batch_size, _, _, depth = target.shape
        c = self.c[None, :, :, None].repeat(batch_size, 1, 1, depth).reshape(-1)
        visible_mask = (target != ignore_index).reshape(-1).nonzero().squeeze(-1)
        if weight is not None:
            weight_mask = weight[None, :] * c[visible_mask, None]
        else:
            weight_mask = c[visible_mask, None]

        num_classes = pred.size(1)
        pred = pred.permute(0, 2, 3, 4, 1).reshape(-1, num_classes)[visible_mask]
        target = target.reshape(-1)[visible_mask]

        if pred.is_cuda:
            calculate_loss_func = sigmoid_focal_loss
        else:
            target = F.one_hot(target, num_classes=num_classes + 1)
            target = target[:, :num_classes]
            calculate_loss_func = py_sigmoid_focal_loss

        return self.loss_weight * calculate_loss_func(
            pred, target.to(torch.long), weight_mask,
            gamma=self.gamma, alpha=self.alpha)
