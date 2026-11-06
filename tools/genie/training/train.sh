#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
LOWORLD_ROOT="${LOWORLD_ROOT:-${PROJECT_DIR:-${PROJECT_ROOT}}}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-${LOWORLD_ROOT}/ckpts/geniedrive/GenieDrive/diffusion_pytorch_model.safetensors}"
SPLIT="${SPLIT:-train}"
NUM_CLIPS="${NUM_CLIPS:-700}"
NUM_FRAMES="${NUM_FRAMES:-81}"
RUN_ROOT="${RUN_ROOT:-${LOWORLD_ROOT}/work_dirs/genie_occ_${SPLIT}${NUM_CLIPS}_${NUM_FRAMES}f}"

export PROJECT_DIR="${PROJECT_DIR:-${LOWORLD_ROOT}}"
export CONDA_ENV="${CONDA_ENV:-loworld-genie}"
export DATASET_DIR="${DATASET_DIR:-${RUN_ROOT}/genie_dataset}"
export META_PATH="${META_PATH:-${DATASET_DIR}/meta.json}"
export OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/finetune_3500steps}"
export LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
export TRANSFORMER_PATH="${TRANSFORMER_PATH:-${INIT_CHECKPOINT}}"

export VIDEO_SAMPLE_N_FRAMES="${VIDEO_SAMPLE_N_FRAMES:-${NUM_FRAMES}}"
export VIDEO_SAMPLE_STRIDE="${VIDEO_SAMPLE_STRIDE:-1}"
export VIDEO_SAMPLE_SIZE="${VIDEO_SAMPLE_SIZE:-512}"
export IMAGE_SAMPLE_SIZE="${IMAGE_SAMPLE_SIZE:-512}"
export TOKEN_SAMPLE_SIZE="${TOKEN_SAMPLE_SIZE:-512}"
export FIX_SAMPLE_H="${FIX_SAMPLE_H:-${HEIGHT:-256}}"
export FIX_SAMPLE_W="${FIX_SAMPLE_W:-${WIDTH:-512}}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
export GPUS="${GPUS:-0}"
export NUM_PROCESSES="${NUM_PROCESSES:-$(awk -F, '{print NF}' <<< "${GPUS}")}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-23600}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-3500}"
export CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-500}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
export LEARNING_RATE="${LEARNING_RATE:-5e-6}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.05}"
export DISABLE_CONDITION_DROPOUT="${DISABLE_CONDITION_DROPOUT:-1}"
export DEPTH_CONTROL_LATENT_WEIGHT="${DEPTH_CONTROL_LATENT_WEIGHT:-0}"
export LOW_VRAM="${LOW_VRAM:-0}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-config/zero_stage2_config.json}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if (( (VIDEO_SAMPLE_N_FRAMES - 1) % 4 != 0 )); then
  echo "VIDEO_SAMPLE_N_FRAMES must satisfy 4k+1 for the Wan VAE; got ${VIDEO_SAMPLE_N_FRAMES}" >&2
  exit 2
fi
if (( FIX_SAMPLE_H % 16 != 0 || FIX_SAMPLE_W % 16 != 0 )); then
  echo "HEIGHT and WIDTH must be divisible by 16; got ${FIX_SAMPLE_H}x${FIX_SAMPLE_W}" >&2
  exit 2
fi

export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PYTHONPATH="${LOWORLD_ROOT}/tools/genie/training/accelerate_compat${PYTHONPATH:+:${PYTHONPATH}}"

cd "${LOWORLD_ROOT}"

if [[ -z "${TRANSFORMER_PATH}" || ! -f "${TRANSFORMER_PATH}" ]]; then
  echo "Initial transformer checkpoint is missing: ${TRANSFORMER_PATH}" >&2
  echo "Set INIT_CHECKPOINT=/path/to/diffusion_pytorch_model.safetensors" >&2
  exit 2
fi
if [[ ! -f "${META_PATH}" ]]; then
  echo "Training metadata is missing: ${META_PATH}" >&2
  echo "Prepare the training conditions with tools/genie/prepare_conditions.sh first." >&2
  exit 2
fi

exec bash "${LOWORLD_ROOT}/tools/genie/training/launch_train.sh"
