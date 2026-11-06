# GenieDrive video runtime

This directory vendors the VideoX-Fun/Wan model and pipeline code used by
LO-World for GenieDrive training and inference. The entrypoints are
[`tools/genie/inference/infer.py`](../../tools/genie/inference/infer.py) and
[`tools/genie/training/train.sh`](../../tools/genie/training/train.sh).
Training data preparation is in `tools/genie/prepare_conditions.sh`.

The inference runtime snapshot comes from source revision
`310c7ac79da0d662359480efd3fb7ae08250c627`; the custom training script supports
LO-World conditions. The vendored package also
contains model families and helpers outside these entrypoints' direct use.

The vendored VideoX-Fun code is distributed under Apache-2.0. Preserve the
upstream license in [`LICENSE`](LICENSE) and see [`README_UPSTREAM.md`](README_UPSTREAM.md)
for the original project information. Model weights have separate terms; follow
the Wan2.1-Fun model page before downloading or redistributing them.
