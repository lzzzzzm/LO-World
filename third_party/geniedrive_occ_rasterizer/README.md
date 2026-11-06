# Vendored GenieDrive occupancy rasterizer

This directory contains the minimal GenieDrive runtime used to turn LO-World
occupancy grids into six-view semantic controls. It removes the runtime
dependency on a separate GenieDrive checkout.

The Python renderer is covered by `GENIEDRIVE_LICENSE` (MIT). The CUDA
rasterizer retains its upstream `diff-gaussian-rasterization/LICENSE.md`.

Install the CUDA extension inside the repository conda environment:

```bash
conda run -n loworld pip install \
  ./third_party/geniedrive_occ_rasterizer/diff-gaussian-rasterization
```

The extension must be compiled for the active PyTorch and CUDA versions; no
prebuilt `.so`, build directory, or environment-specific metadata is vendored.
