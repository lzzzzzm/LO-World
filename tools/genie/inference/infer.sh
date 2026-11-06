#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-loworld-genie}"
GPU_IDS="${GPU_IDS:-${GPU:-0}}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR to a prepared Genie condition dataset}"
CHECKPOINT="${CHECKPOINT:-${ROOT}/ckpts/geniedrive/geniedrive_448x800.safetensors}"
BASE_MODEL="${BASE_MODEL:-${ROOT}/ckpts/geniedrive/Wan2.1-Fun-V1.1-1.3B-Control}"
SAVE_DIR="${SAVE_DIR:-${ROOT}/work_dirs/genie_inference}"
GENIE_DIR="${ROOT}/third_party/geniedrive_occ_render"
META_PATH="${META_PATH:-${DATA_DIR}/meta.json}"
IFS=',' read -r -a gpu_ids <<< "${GPU_IDS}"
NUM_GPUS="${NUM_GPUS:-${#gpu_ids[@]}}"

if [[ "${NUM_GPUS}" -ne "${#gpu_ids[@]}" ]]; then
  echo "NUM_GPUS=${NUM_GPUS} does not match GPU_IDS=${GPU_IDS}" >&2
  exit 2
fi

if [[ ! -d "${BASE_MODEL}" ]]; then
  echo "Wan2.1-Fun base model not found: ${BASE_MODEL}" >&2
  echo "Download it as described in README.md#downstream-video-generation." >&2
  exit 2
fi
if [[ ! -f "${META_PATH}" ]]; then
  echo "Condition manifest not found: ${META_PATH}" >&2
  exit 2
fi

args=(
  --genie-dir "${GENIE_DIR}"
  --base-model "${BASE_MODEL}"
  --checkpoint "${CHECKPOINT}"
  --data-dir "${DATA_DIR}"
  --meta-path "${META_PATH}"
  --save-dir "${SAVE_DIR}"
  --height "${HEIGHT:-448}"
  --width "${WIDTH:-800}"
  --video-length "${NUM_FRAMES:-17}"
  --fps "${FPS:-12}"
  --num-inference-steps "${NUM_INFERENCE_STEPS:-50}"
  --guidance-scale "${GUIDANCE_SCALE:-6.0}"
  --dtype "${DTYPE:-bf16}"
  --gpu-memory-mode "${GPU_MEMORY_MODE:-model_cpu_offload}"
  --max-samples "${MAX_SAMPLES:-1}"
)
if [[ -n "${START_INDEX:-}" ]]; then
  args+=(--start-index "${START_INDEX}")
fi

if [[ "${OVERWRITE:-0}" == "1" ]]; then
  args+=(--overwrite)
fi
if [[ "${SAVE_GRID:-0}" == "1" ]]; then
  args+=(--save-grid)
fi
if [[ "${DISABLE_TEACACHE:-0}" == "1" ]]; then
  args+=(--disable-teacache)
fi

cd "${ROOT}"
if [[ "${NUM_GPUS}" -eq 1 ]]; then
  launcher=(python tools/genie/inference/infer.py)
else
  launcher=(torchrun --standalone --nproc_per_node="${NUM_GPUS}" tools/genie/inference/infer.py)
fi
exec conda run --no-capture-output -n "${CONDA_ENV}" env \
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
  PYTHONPATH="${ROOT}:${GENIE_DIR}" \
  "${launcher[@]}" "${args[@]}"
