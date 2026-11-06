import logging
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader

from mmengine.logging import print_log
from mmengine.registry import LOOPS
from mmengine.runner.base_loop import BaseLoop
from mmengine.runner.loops import _InfiniteDataloaderIterator
from mmengine.runner.utils import calc_dynamic_intervals


@LOOPS.register_module()
class SequenceIterBasedTrainLoop(BaseLoop):
    """Iter-based loop with epoch-compatible virtual boundaries.

    The dataloader iterator is kept alive for the whole training run. This is
    required by sequence batch samplers whose per-batch-slot buffers should
    continue as idx, idx + 1, ... until a sequence group is exhausted.
    """

    def __init__(
            self,
            runner,
            dataloader: Union[DataLoader, Dict],
            max_epochs: Optional[int] = None,
            max_iters: Optional[int] = None,
            iters_per_epoch: Optional[int] = None,
            val_begin: int = 1,
            val_interval: int = 1,
            dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        super().__init__(runner, dataloader)

        if iters_per_epoch is None:
            iters_per_epoch = len(self.dataloader)
        if iters_per_epoch <= 0:
            raise ValueError(
                f'iters_per_epoch must be positive, got {iters_per_epoch}')

        if max_iters is None:
            if max_epochs is None:
                raise ValueError(
                    'SequenceIterBasedTrainLoop requires max_epochs or '
                    'max_iters.')
            max_iters = int(max_epochs) * int(iters_per_epoch)

        if max_epochs is None:
            max_epochs = (int(max_iters) + int(iters_per_epoch) - 1) // int(
                iters_per_epoch)

        self._max_epochs = int(max_epochs)
        self._max_iters = int(max_iters)
        self.iters_per_epoch = int(iters_per_epoch)
        self._epoch = 0
        self._iter = 0
        self.val_begin = val_begin
        self.val_interval = val_interval
        self.stop_training = False

        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in visualizer will be None.',
                logger='current',
                level=logging.WARNING)

        self.dataloader_iterator = _InfiniteDataloaderIterator(self.dataloader)
        self.dynamic_milestones, self.dynamic_intervals = \
            calc_dynamic_intervals(self.val_interval, dynamic_intervals)

    @property
    def max_epochs(self):
        return self._max_epochs

    @property
    def max_iters(self):
        return self._max_iters

    @property
    def epoch(self):
        return self._epoch

    @property
    def iter(self):
        return self._iter

    def run(self) -> torch.nn.Module:
        self.runner.call_hook('before_train')

        while self._iter < self._max_iters and not self.stop_training:
            self.runner.call_hook('before_train_epoch')
            self.runner.model.train()
            epoch_end_iter = min(self._iter + self.iters_per_epoch,
                                 self._max_iters)

            while self._iter < epoch_end_iter and not self.stop_training:
                data_batch = next(self.dataloader_iterator)
                self.run_iter(self._iter % self.iters_per_epoch, data_batch)

            self.runner.call_hook('after_train_epoch')
            self._epoch += 1
            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._epoch >= self.val_begin
                    and (self._epoch % self.val_interval == 0
                         or self._iter == self._max_iters)):
                self.runner.val_loop.run()

        self.runner.call_hook('after_train')
        return self.runner.model

    def run_iter(self, batch_idx: int, data_batch: Sequence[dict]) -> None:
        self.runner.call_hook(
            'before_train_iter', batch_idx=batch_idx, data_batch=data_batch)
        outputs = self.runner.model.train_step(
            data_batch, optim_wrapper=self.runner.optim_wrapper)
        self.runner.call_hook(
            'after_train_iter',
            batch_idx=batch_idx,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1

    def _decide_current_val_interval(self) -> None:
        import bisect

        step = bisect.bisect(self.dynamic_milestones, (self.epoch + 1))
        self.val_interval = self.dynamic_intervals[step - 1]
