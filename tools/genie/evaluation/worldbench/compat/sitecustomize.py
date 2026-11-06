"""Process-local compatibility fixes for legacy MMCV on modern PyTorch."""

from torch.nn.parallel import DistributedDataParallel


# MMCV 1.6 overrides _run_ddp_forward using an attribute that was removed
# from PyTorch's DistributedDataParallel in newer releases.
if not hasattr(DistributedDataParallel, "_use_replicated_tensor_module"):
    DistributedDataParallel._use_replicated_tensor_module = False
