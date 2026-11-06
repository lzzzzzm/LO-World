# Troubleshooting

## MMCV Installation

If MMCV installation fails, build it from source in the `loworld` environment.
Install PyTorch and a matching CUDA Toolkit first, as described in the
[installation guide](../README.md#lo-world-tokenizer-and-occupancy-world-model).

```bash
conda activate loworld
git clone https://github.com/open-mmlab/mmcv.git
mv mmcv mmcv_source
cd mmcv_source
git checkout v2.1.0
env MMCV_WITH_OPS=1 pip install -e . --no-build-isolation
```

This installs MMCV 2.1.0 into the active environment.
