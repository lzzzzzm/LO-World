#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-loworld}"
SPLIT="${SPLIT:-train}"
NUM_CLIPS="${NUM_CLIPS:-700}"
NUM_FRAMES="${NUM_FRAMES:-81}"
DATASET_DIR="${DATASET_DIR:-${ROOT}/work_dirs/genie_pred_occ_${SPLIT}${NUM_CLIPS}_${NUM_FRAMES}f}"
CONDITION_HEIGHT="${CONDITION_HEIGHT:-450}"
CONDITION_WIDTH="${CONDITION_WIDTH:-800}"
FPS="${FPS:-12}"
GPUS="${GPUS:-0}"
IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
NUM_SHARDS="${#GPU_IDS[@]}"
LOG_DIR="${DATASET_DIR}/logs"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-5}"

mkdir -p "${LOG_DIR}"
cd "${ROOT}"

TOTAL_CLIPS="$(conda run -n "${CONDA_ENV}" python -c \
  "import json; print(len(json.load(open('${DATASET_DIR}/selection_manifest.json'))['records']))")"
PROGRESS_DIR="$(mktemp -d "${LOG_DIR}/.video_progress.XXXXXX")"
trap 'rm -rf -- "${PROGRESS_DIR}"' EXIT
echo "Video export status: total=${TOTAL_CLIPS}; checking existing outputs on ${NUM_SHARDS} shards"
echo "Requested semantic resolution=${CONDITION_HEIGHT}x${CONDITION_WIDTH}, fps=${FPS}; incompatible old manifests will be regenerated"

pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${GPU_IDS[$shard]}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="${ROOT}" \
  conda run --no-capture-output -n "${CONDA_ENV}" python \
    "${ROOT}/tools/genie/video_export/build_dataset.py" \
    --dataset-dir "${DATASET_DIR}" \
    --height "${CONDITION_HEIGHT}" \
    --width "${CONDITION_WIDTH}" \
    --fps "${FPS}" \
    --progress-dir "${PROGRESS_DIR}" \
    --num-shards "${NUM_SHARDS}" \
    --shard-id "${shard}" \
    > "${LOG_DIR}/genie_export_shard${shard}.log" 2>&1 &
  pids+=("$!")
done

while :; do
  active=0
  for pid in "${pids[@]}"; do
    kill -0 "${pid}" 2>/dev/null && active=$((active + 1))
  done
  reused="$(find "${PROGRESS_DIR}" -maxdepth 1 -type f -name 'reused_*' | wc -l)"
  generated="$(find "${PROGRESS_DIR}" -maxdepth 1 -type f -name 'generated_*' | wc -l)"
  complete=$((reused + generated))
  percent=$((TOTAL_CLIPS > 0 ? complete * 100 / TOTAL_CLIPS : 100))
  printf '\rVideo export: %d/%d (%d%%) | reused=%d | generated=%d | active_shards=%d' \
    "${complete}" "${TOTAL_CLIPS}" "${percent}" "${reused}" "${generated}" "${active}"
  (( active == 0 )) && break
  sleep "${PROGRESS_INTERVAL}"
done
printf '\n'

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "Video export failed: reused=${reused}, generated=${generated}; see ${LOG_DIR}" >&2
  exit 1
fi
echo "Video export summary: reused=${reused}, generated=${generated}, total=${TOTAL_CLIPS}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${ROOT}" \
conda run --no-capture-output -n "${CONDA_ENV}" python \
  "${ROOT}/tools/genie/video_export/build_dataset.py" \
  --dataset-dir "${DATASET_DIR}" \
  --height "${CONDITION_HEIGHT}" \
  --width "${CONDITION_WIDTH}" \
  --fps "${FPS}" \
  --finalize-only \
  > "${LOG_DIR}/genie_export_finalize.log" 2>&1

echo "Completed Genie training dataset: ${DATASET_DIR}/genie_dataset"
