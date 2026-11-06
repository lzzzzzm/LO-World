import copy
import itertools
import math
from typing import Iterator, Optional

import numpy as np
import torch
import torch.distributed as dist
from mmengine.registry import DATA_SAMPLERS
from torch.utils.data import Sampler


@DATA_SAMPLERS.register_module()
class SequenceGroupBatchSampler(Sampler):
    """Yield batches that advance each sample slot through one sequence.

    The dataset must expose flag where equal consecutive values belong to the
    same scene or scene split. Group order is shuffled, but indices inside each
    group are kept sorted, so a sample slot sees idx, idx + 1, ... until that
    group is exhausted.
    """

    def __init__(self,
                 sampler: Sampler,
                 batch_size: int = 1,
                 shuffle: bool = True,
                 seed: Optional[int] = None,
                 drop_last: bool = False) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self.sampler = sampler
        self.dataset = sampler.dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rank = getattr(sampler, "rank", 0)
        self.world_size = getattr(sampler, "world_size", 1)
        self.seed = getattr(sampler, "seed", 0) if seed is None else seed
        if self.seed is None:
            self.seed = 0
        self.epoch = 0

        if not hasattr(self.dataset, "flag") or self.dataset.flag is None:
            raise AttributeError(
                "SequenceGroupBatchSampler requires dataset.flag. "
                "Set use_sequence_group_flag=True in dataset config.")

        self.flag = np.asarray(self.dataset.flag, dtype=np.int64)
        if len(self.flag) != len(self.dataset):
            raise ValueError(
                "dataset.flag length must match dataset length: "
                f"{len(self.flag)} vs {len(self.dataset)}")

        self.group_ids = np.unique(self.flag).tolist()
        self.group_to_indices = {
            group_id: np.where(self.flag == group_id)[0].tolist()
            for group_id in self.group_ids
        }
        self.global_batch_size = self.batch_size * self.world_size
        if len(self.group_ids) < self.global_batch_size:
            raise ValueError(
                "SequenceGroupBatchSampler requires at least one sequence "
                "group per global sample slot: "
                f"{len(self.group_ids)} groups vs global batch size "
                f"{self.global_batch_size}. Increase sequences_split_num or "
                "reduce batch_size/world_size.")

    def _infinite_group_ids(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        group_ids = list(self.group_ids)
        while True:
            if self.shuffle:
                order = torch.randperm(len(group_ids), generator=generator).tolist()
                for idx in order:
                    yield group_ids[idx]
            else:
                yield from group_ids

    def _group_ids_for_slot(self, global_slot_idx: int) -> Iterator[int]:
        yield from itertools.islice(
            self._infinite_group_ids(),
            global_slot_idx,
            None,
            self.global_batch_size,
        )

    def __iter__(self) -> Iterator[list]:
        group_iterators = [
            self._group_ids_for_slot(self.rank * self.batch_size + local_idx)
            for local_idx in range(self.batch_size)
        ]
        buffers = [[] for _ in range(self.batch_size)]

        while True:
            batch = []
            for local_idx in range(self.batch_size):
                if len(buffers[local_idx]) == 0:
                    group_id = next(group_iterators[local_idx])
                    buffers[local_idx] = copy.deepcopy(self.group_to_indices[group_id])
                batch.append(buffers[local_idx].pop(0))

            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / max(1, self.global_batch_size))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)


@DATA_SAMPLERS.register_module()
class SequenceGroupDistributedSampler(Sampler):
    """Distributed eval sampler that keeps each sequence on one rank.

    BEVWorld keeps a per-sample-slot memory during prediction. Splitting one
    scene across ranks breaks that memory because a rank may start from the
    middle of a sequence. This sampler assigns complete dataset.flag groups to
    ranks and preserves the original index order inside every group.
    """

    def __init__(self,
                 dataset,
                 shuffle: bool = False,
                 seed: Optional[int] = None,
                 rank: Optional[int] = None,
                 world_size: Optional[int] = None) -> None:
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = 0 if seed is None else seed
        self.epoch = 0

        if rank is None or world_size is None:
            if dist.is_available() and dist.is_initialized():
                dist_rank = dist.get_rank()
                dist_world_size = dist.get_world_size()
            else:
                dist_rank = 0
                dist_world_size = 1
            self.rank = dist_rank if rank is None else rank
            self.world_size = (
                dist_world_size if world_size is None else world_size)
        else:
            self.rank = rank
            self.world_size = world_size

        if self.world_size <= 0:
            raise ValueError(
                f"world_size must be positive, got {self.world_size}")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(
                f"rank must be in [0, {self.world_size}), got {self.rank}")

        if not hasattr(self.dataset, "flag") or self.dataset.flag is None:
            raise AttributeError(
                "SequenceGroupDistributedSampler requires dataset.flag. "
                "Set use_sequence_group_flag=True in dataset config.")

        self.flag = np.asarray(self.dataset.flag, dtype=np.int64)
        if len(self.flag) != len(self.dataset):
            raise ValueError(
                "dataset.flag length must match dataset length: "
                f"{len(self.flag)} vs {len(self.dataset)}")

        self.group_ids = np.unique(self.flag).tolist()
        self.group_to_indices = {
            group_id: np.where(self.flag == group_id)[0].tolist()
            for group_id in self.group_ids
        }
        if len(self.group_ids) < self.world_size:
            raise ValueError(
                "SequenceGroupDistributedSampler requires at least one "
                "sequence group per rank: "
                f"{len(self.group_ids)} groups vs world_size "
                f"{self.world_size}.")

    def _ordered_group_ids(self) -> list:
        group_ids = list(self.group_ids)
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            order = torch.randperm(len(group_ids), generator=generator).tolist()
            group_ids = [group_ids[idx] for idx in order]
        return group_ids

    def _rank_group_ids(self) -> list:
        rank_group_ids = [[] for _ in range(self.world_size)]
        rank_lengths = [0 for _ in range(self.world_size)]

        for group_id in self._ordered_group_ids():
            target_rank = min(range(self.world_size),
                              key=lambda idx: rank_lengths[idx])
            rank_group_ids[target_rank].append(group_id)
            rank_lengths[target_rank] += len(self.group_to_indices[group_id])

        return rank_group_ids[self.rank]

    def _rank_indices(self) -> list:
        indices = []
        for group_id in self._rank_group_ids():
            indices.extend(self.group_to_indices[group_id])
        return indices

    def __iter__(self) -> Iterator[int]:
        yield from self._rank_indices()

    def __len__(self) -> int:
        return len(self._rank_indices())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
