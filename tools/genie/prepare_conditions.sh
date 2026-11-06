#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-loworld}"
SPLIT="${SPLIT:-val}"
NUM_CLIPS="${NUM_CLIPS:-150}"
NUM_FRAMES="${NUM_FRAMES:-81}"
FUTURE_FRAMES="${FUTURE_FRAMES:-$((NUM_FRAMES - 1))}"
DATASET_DIR="${DATASET_DIR:-${ROOT}/work_dirs/genie_occ_${SPLIT}${NUM_CLIPS}_${NUM_FRAMES}f}"
GPUS="${GPUS:-0}"
CONDITION_HEIGHT="${CONDITION_HEIGHT:-450}"
CONDITION_WIDTH="${CONDITION_WIDTH:-800}"
FPS="${FPS:-12}"
SAVE_POINTS="${SAVE_POINTS:-0}"

if (( NUM_FRAMES < 2 || FUTURE_FRAMES != NUM_FRAMES - 1 )); then
  echo "NUM_FRAMES must be >= 2 and FUTURE_FRAMES must equal NUM_FRAMES - 1" >&2
  exit 2
fi

export ROOT CONDA_ENV SPLIT NUM_CLIPS NUM_FRAMES FUTURE_FRAMES DATASET_DIR SAVE_POINTS
export GPUS CONDITION_HEIGHT CONDITION_WIDTH FPS
cd "${ROOT}"

echo "[1/3] Preparing ${NUM_CLIPS} ${SPLIT} clips (${NUM_FRAMES} frames each)"
bash tools/genie/occ_generation/prepare_dataset.sh "$@"

echo "[2/3] Generating future OCC with ${CONDA_ENV} on GPU(s) ${GPUS}"
bash tools/genie/occ_generation/generate.sh

echo "[3/3] Rendering GenieDrive video conditions at ${CONDITION_HEIGHT}x${CONDITION_WIDTH}, ${FPS} fps"
bash tools/genie/video_export/build.sh

echo "Ready for GenieDrive inference: ${DATASET_DIR}/genie_dataset"
