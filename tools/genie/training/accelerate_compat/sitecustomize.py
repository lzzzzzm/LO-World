"""Bind CUDA before Accelerate's DeepSpeed branch initializes NCCL.

Accelerate in the loworld environment calls DeepSpeed distributed init
before its normal ``set_device`` step.  PyTorch 2.4 eagerly initializes NCCL
and otherwise sees every local rank on cuda:0.  This directory is prepended
to PYTHONPATH only by the 100-clip training wrapper.
"""

from __future__ import annotations

import os


local_rank = os.environ.get("LOCAL_RANK")
if local_rank is not None and int(local_rank) >= 0:
    import torch

    if torch.cuda.is_available():
        torch.cuda.set_device(int(local_rank))
