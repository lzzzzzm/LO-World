#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data/nuscenes}"
TRAIN_ANN="${TRAIN_ANN:-data/nus_infos/nuscenes_infos_train.pkl}"
VAL_ANN="${VAL_ANN:-data/nus_infos/nuscenes_infos_val.pkl}"
OUT_ROOT="${OUT_ROOT:-data/nus_tokenizer_lidar_range}"
WORKERS="${WORKERS:-32}"
PYTHON_BIN="${PYTHON_BIN:-python}"

COMMON_ARGS=(
  --data-root "${DATA_ROOT}"
  --pc-range -40 -40 -1 40 40 5.4
  --workers "${WORKERS}"
  --overwrite
)

echo "Precomputing train range images -> ${OUT_ROOT}"
PYTHONPATH=. "${PYTHON_BIN}" tools/precompute_nuscenes_lidar_range.py \
  "${TRAIN_ANN}" \
  "${OUT_ROOT}" \
  "${COMMON_ARGS[@]}"

echo "Precomputing val range images -> ${OUT_ROOT}"
PYTHONPATH=. "${PYTHON_BIN}" tools/precompute_nuscenes_lidar_range.py \
  "${VAL_ANN}" \
  "${OUT_ROOT}" \
  "${COMMON_ARGS[@]}"

echo "Done. Range images saved to ${OUT_ROOT}"
