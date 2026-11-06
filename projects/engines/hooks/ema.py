import math
import os
from copy import deepcopy

import torch
from torch import nn

from mmengine.hooks import Hook
from mmengine.logging import print_log
from mmengine.dist import master_only
from mmdet3d.registry import HOOKS
from mmengine.model import is_model_wrapper

class ModelEMA:
    """Model Exponential Moving Average.

    Keep a moving average of everything in the model state_dict (parameters and
    buffers).
    """

    def __init__(self, model, decay=0.9999, updates=0):
        """
        Args:
            model (nn.Module): model to apply EMA.
            decay (float): ema decay rate.
            updates (int): counter of EMA updates.
        """
        # Create EMA(FP32)
        self.ema = deepcopy(model).eval()
        if is_model_wrapper(self.ema):
            self.ema = self.ema.module

        self.updates = updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            if is_model_wrapper(model):
                msd = model.module.state_dict()
            else:
                msd = model.state_dict()

            for k, v in self.ema.state_dict().items():
                if 'vqvae.' in k or 'vae.' in k:
                    # Frozen modules are excluded from EMA averaging, but they
                    # still need the live model's state. In particular, the
                    # EMA copy can be created before MMEngine restores
                    # ``load_from``; skipping these keys would then preserve
                    # their random initialization in every EMA checkpoint.
                    v.copy_(msd[k].detach())
                    continue
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1.0 - d) * msd[k].detach()


@HOOKS.register_module()
class MEGVIIEMAHook(Hook):
    """EMAHook used in BEVDepth.

    Modified from https://github.com/Megvii-Base
    Detection/BEVDepth/blob/main/callbacks/ema.py.
    """

    def __init__(self, init_updates=0, decay=0.9990, resume=None, interval=-1, max_keep_ckpts=-1, by_epoch=True):
        super().__init__()
        self.init_updates = init_updates
        self.resume = resume
        self.decay = decay
        self.interval = interval
        self.max_keep_ckpts = max_keep_ckpts
        self.by_epoch = by_epoch
        self.keep_ckpt_paths = []
        self.keep_iter_ckpt_paths = []

    def before_run(self, runner):
        runner.ema_model = ModelEMA(runner.model, self.decay)
        runner.ema_model.updates = self.init_updates

        if self.resume is not None:
            runner.logger.info(f'resume ema checkpoint from {self.resume}')
            cpt = torch.load(self.resume, map_location='cpu', weights_only=True)
            runner.ema_model.ema.load_state_dict(cpt['state_dict'])
            runner.ema_model.updates = cpt['updates']

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        runner.ema_model.update(runner.model)
        if not self.by_epoch:
            curr_step = runner.iter
            if self.interval > 0:
                if curr_step % self.interval == 0 and curr_step > 0:
                    self.save_checkpoint_iter(runner)

    def after_run(self, runner):
        # Only save if it was a training run (runner.iter > 0)
        if runner.iter > 0:
            if self.by_epoch:
                self.save_checkpoint(runner)
            else:
                self.save_checkpoint_iter(runner)

    @master_only
    def after_train_epoch(self, runner):
        if self.by_epoch:
            if self.interval > 0 and (runner.epoch + 1) % self.interval != 0:
                return
            self.save_checkpoint(runner)

    @master_only
    def save_checkpoint(self, runner):
        state_dict = runner.ema_model.ema.state_dict()
        ema_checkpoint = {
            'epoch': runner.epoch,
            'state_dict': state_dict,
            'updates': runner.ema_model.updates
        }
        if runner.epoch is not None:
            save_path = f'epoch_{runner.epoch + 1}_ema.pth'
        else:
            save_path = 'ema.pth'
        save_path = os.path.join(runner.work_dir, save_path)
        torch.save(ema_checkpoint, save_path)
        runner.logger.info(f'Saving ema checkpoint at {save_path}')

        if self.max_keep_ckpts > 0:
            self.keep_ckpt_paths.append(save_path)
            if len(self.keep_ckpt_paths) > self.max_keep_ckpts:
                to_delete = self.keep_ckpt_paths.pop(0)
                if os.path.exists(to_delete):
                    os.remove(to_delete)

    @master_only
    def save_checkpoint_iter(self, runner):
        state_dict = runner.ema_model.ema.state_dict()
        ema_checkpoint = {
            'iter': runner.iter,
            'state_dict': state_dict,
            'updates': runner.ema_model.updates
        }
        save_path = f'iter_{runner.iter}_ema.pth'
        save_path = os.path.join(runner.work_dir, save_path)
        torch.save(ema_checkpoint, save_path)
        runner.logger.info(f'Saving ema checkpoint at {save_path}')

        if self.max_keep_ckpts > 0:
            self.keep_iter_ckpt_paths.append(save_path)
            if len(self.keep_iter_ckpt_paths) > self.max_keep_ckpts:
                to_delete = self.keep_iter_ckpt_paths.pop(0)
                if os.path.exists(to_delete):
                    os.remove(to_delete)
