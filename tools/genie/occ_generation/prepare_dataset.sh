#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-loworld}"
SPLIT="${SPLIT:-train}"
NUM_CLIPS="${NUM_CLIPS:-700}"
NUM_FRAMES="${NUM_FRAMES:-81}"
FUTURE_FRAMES="${FUTURE_FRAMES:-$((NUM_FRAMES - 1))}"
DATASET_DIR="${DATASET_DIR:-${ROOT}/work_dirs/genie_pred_occ_${SPLIT}${NUM_CLIPS}_${NUM_FRAMES}f}"
LO_ANN_FILE="${LO_ANN_FILE:-${ROOT}/data/nus_infos/nuscenes_infos_${SPLIT}.pkl}"
ANN_12HZ="${ANN_12HZ:-${ROOT}/data/nus_infos_12hz/nuscenes_interp_12Hz_infos_${SPLIT}.pkl}"

if (( NUM_FRAMES < 2 || FUTURE_FRAMES != NUM_FRAMES - 1 )); then
  echo "NUM_FRAMES must be >= 2 and FUTURE_FRAMES must equal NUM_FRAMES - 1" >&2
  exit 2
fi

cd "${ROOT}"
PREPARED_LAYOUT_DIR="${PREPARED_LAYOUT_DIR:-${ROOT}/data/bev_layout/12hz_${SPLIT}${NUM_CLIPS}_${NUM_FRAMES}f}"
if [[ "${USE_PREPARED_LAYOUTS:-1}" == "1" && $# -eq 0 \
      && "${LO_ANN_FILE}" == "${ROOT}/data/nus_infos/nuscenes_infos_${SPLIT}.pkl" \
      && "${ANN_12HZ}" == "${ROOT}/data/nus_infos_12hz/nuscenes_interp_12Hz_infos_${SPLIT}.pkl" \
      && -f "${PREPARED_LAYOUT_DIR}/selection_manifest.json" ]]; then
  exec conda run --no-capture-output -n "${CONDA_ENV}" python \
    tools/genie/occ_generation/prepared_conditions.py \
    --source "${PREPARED_LAYOUT_DIR}" --out-dir "${DATASET_DIR}" \
    --split "${SPLIT}" --num-clips "${NUM_CLIPS}" --num-frames "${NUM_FRAMES}"
fi
exec conda run --no-capture-output -n "${CONDA_ENV}" python \
  tools/genie/occ_generation/prepare_dataset.py \
  --lo-ann-file "${LO_ANN_FILE}" \
  --ann-12hz "${ANN_12HZ}" \
  --out-dir "${DATASET_DIR}" \
  --split "${SPLIT}" \
  --num-clips "${NUM_CLIPS}" \
  --future-frames "${FUTURE_FRAMES}" \
  --allow-scene-repeats \
  "$@"
