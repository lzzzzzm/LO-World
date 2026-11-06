#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-loworld}"
SPLIT="${SPLIT:-train}"
NUM_CLIPS="${NUM_CLIPS:-700}"
NUM_FRAMES="${NUM_FRAMES:-81}"
DATASET_DIR="${DATASET_DIR:-${ROOT}/work_dirs/genie_pred_occ_${SPLIT}${NUM_CLIPS}_${NUM_FRAMES}f}"
CONFIG="${CONFIG:-${ROOT}/configs/world_model/bev_world_12f_layout_autoregressive.py}"
CHECKPOINT="${CHECKPOINT:-${ROOT}/ckpts/bev_world_12f_layout_autoregressive.pth}"
GPU_IDS_CSV="${GPUS:-0}"
IFS=',' read -r -a GPU_IDS <<< "${GPU_IDS_CSV}"
NUM_SHARDS="${NUM_SHARDS:-${#GPU_IDS[@]}}"
LOG_DIR="${DATASET_DIR}/logs"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-5}"
SAVE_POINTS="${SAVE_POINTS:-0}"

GENERATION_FLAGS=()
if [[ "${SAVE_POINTS}" == "1" ]]; then
  GENERATION_FLAGS+=(--save-points)
fi

if [[ "${NUM_SHARDS}" -ne "${#GPU_IDS[@]}" ]]; then
  echo "NUM_SHARDS must match the number of GPUS (${#GPU_IDS[@]})" >&2
  exit 2
fi

if [[ ! -f "${DATASET_DIR}/selection_manifest.json" ]]; then
  echo "Missing ${DATASET_DIR}/selection_manifest.json" >&2
  echo "Run tools/genie/occ_generation/prepare_dataset.sh first." >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"
cd "${ROOT}"

TOTAL_CLIPS="$(conda run -n "${CONDA_ENV}" python -c \
  "import json; print(len(json.load(open('${DATASET_DIR}/selection_manifest.json'))['records']))")"
PROGRESS_DIR="$(mktemp -d "${LOG_DIR}/.occ_progress.XXXXXX")"
trap 'rm -rf -- "${PROGRESS_DIR}"' EXIT
echo "OCC status: total=${TOTAL_CLIPS}; checking existing outputs on ${NUM_SHARDS} shards"

echo "Generate ${NUM_FRAMES}-frame LO-World predictions (${NUM_SHARDS} GPUs; SAVE_POINTS=${SAVE_POINTS})"
pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${GPU_IDS[$shard]}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="${ROOT}" \
  conda run --no-capture-output -n "${CONDA_ENV}" python \
    "${ROOT}/tools/genie/occ_generation/generate_batch.py" \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --dataset-dir "${DATASET_DIR}" \
    --dataset-split "${SPLIT}" \
    "${GENERATION_FLAGS[@]}" \
    --progress-dir "${PROGRESS_DIR}" \
    --num-shards "${NUM_SHARDS}" \
    --shard-id "${shard}" \
    > "${LOG_DIR}/predict_shard${shard}.log" 2>&1 &
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
  printf '\rOCC progress: %d/%d (%d%%) | reused=%d | generated=%d | active_shards=%d' \
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
  echo "OCC failed: reused=${reused}, generated=${generated}; see ${LOG_DIR}" >&2
  exit 1
fi
echo "OCC run summary: reused=${reused}, generated=${generated}, total=${TOTAL_CLIPS}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${ROOT}" \
conda run --no-capture-output -n "${CONDA_ENV}" python \
  "${ROOT}/tools/genie/occ_generation/generate_batch.py" \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --dataset-dir "${DATASET_DIR}" \
  "${GENERATION_FLAGS[@]}" \
  --num-shards "${NUM_SHARDS}" \
  --finalize-only \
  > "${LOG_DIR}/predict_finalize.log" 2>&1

echo "Completed LO-World OCC generation in ${DATASET_DIR}/predictions"
