#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
GENIE_DIR="${GENIE_DIR:-${PROJECT_DIR}/third_party/geniedrive_occ_render}"
CONDA_ENV="${CONDA_ENV:-loworld-genie}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-config/zero_stage2_config.json}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_PATH="${WAN21_FUN_MODEL:-${PROJECT_DIR}/ckpts/geniedrive/Wan2.1-Fun-V1.1-1.3B-Control}"
if [[ -z "${DATASET_DIR:-}" || -z "${OUTPUT_DIR:-}" ]]; then
  echo "ERROR: DATASET_DIR and OUTPUT_DIR are required by this launcher." >&2
  echo "Use bash tools/genie/training/train.sh for the training defaults." >&2
  exit 2
fi
[[ "${DATASET_DIR}" = /* ]] || DATASET_DIR="${PROJECT_DIR}/${DATASET_DIR}"
META_PATH="${META_PATH:-${DATASET_DIR}/meta.json}"
[[ "${META_PATH}" = /* ]] || META_PATH="${PROJECT_DIR}/${META_PATH}"
[[ "${OUTPUT_DIR}" = /* ]] || OUTPUT_DIR="${PROJECT_DIR}/${OUTPUT_DIR}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
[[ "${LOG_DIR}" = /* ]] || LOG_DIR="${PROJECT_DIR}/${LOG_DIR}"

GPUS="${GPUS:-0}"
NUM_PROCESSES="${NUM_PROCESSES:-$(awk -F, '{print NF}' <<< "${GPUS}")}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-23521}"

IMAGE_SAMPLE_SIZE="${IMAGE_SAMPLE_SIZE:-512}"
VIDEO_SAMPLE_SIZE="${VIDEO_SAMPLE_SIZE:-512}"
TOKEN_SAMPLE_SIZE="${TOKEN_SAMPLE_SIZE:-512}"
VIDEO_SAMPLE_STRIDE="${VIDEO_SAMPLE_STRIDE:-1}"
VIDEO_SAMPLE_N_FRAMES="${VIDEO_SAMPLE_N_FRAMES:-17}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
LOW_VRAM="${LOW_VRAM:-0}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-320}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-500}"
LEARNING_RATE="${LEARNING_RATE:-2e-05}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-100}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.05}"
FIX_SAMPLE_H="${FIX_SAMPLE_H:-256}"
FIX_SAMPLE_W="${FIX_SAMPLE_W:-512}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
TRANSFORMER_PATH="${TRANSFORMER_PATH:-}"
DISABLE_CONDITION_DROPOUT="${DISABLE_CONDITION_DROPOUT:-1}"
DEPTH_CONTROL_LATENT_WEIGHT="${DEPTH_CONTROL_LATENT_WEIGHT:-0}"
if [[ -n "${TRANSFORMER_PATH}" && "${TRANSFORMER_PATH}" != /* ]]; then
  TRANSFORMER_PATH="${PROJECT_DIR}/${TRANSFORMER_PATH}"
fi

if [[ ! -e "${MODEL_PATH}" ]]; then
  echo "ERROR: MODEL_PATH does not exist: ${MODEL_PATH}" >&2
  exit 2
fi
if [[ ! -d "${GENIE_DIR}" ]]; then
  echo "ERROR: GENIE_DIR does not exist: ${GENIE_DIR}" >&2
  exit 2
fi
if [[ ! -f "${META_PATH}" ]]; then
  echo "ERROR: meta.json is missing: ${META_PATH}" >&2
  echo "Prepare the training conditions with SPLIT=train NUM_CLIPS=700 bash tools/genie/prepare_conditions.sh first." >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

if [[ -n "${CONDA_SH:-}" ]]; then
  source "${CONDA_SH}"
else
  CONDA_BASE="$(conda info --base)"
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV}"

cd "${GENIE_DIR}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF

EXTRA_STEP_ARGS=()
if [[ -n "${MAX_TRAIN_STEPS}" ]]; then
  EXTRA_STEP_ARGS+=(--max_train_steps "${MAX_TRAIN_STEPS}")
fi
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  EXTRA_STEP_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
if [[ -n "${TRANSFORMER_PATH}" ]]; then
  EXTRA_STEP_ARGS+=(--transformer_path "${TRANSFORMER_PATH}")
fi
if [[ "${LOW_VRAM}" == "1" ]]; then
  EXTRA_STEP_ARGS+=(--low_vram)
fi
if [[ "${DISABLE_CONDITION_DROPOUT}" == "1" ]]; then
  EXTRA_STEP_ARGS+=(--disable_condition_dropout)
fi
if [[ "${DEPTH_CONTROL_LATENT_WEIGHT}" != "0" ]]; then
  EXTRA_STEP_ARGS+=(--depth_control_latent_weight "${DEPTH_CONTROL_LATENT_WEIGHT}")
fi

echo "Logging to ${LOG_FILE}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "DATASET_DIR=${DATASET_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "TRANSFORMER_PATH=${TRANSFORMER_PATH}"
echo "LOW_VRAM=${LOW_VRAM}"
echo "DISABLE_CONDITION_DROPOUT=${DISABLE_CONDITION_DROPOUT}"
echo "DEPTH_CONTROL_LATENT_WEIGHT=${DEPTH_CONTROL_LATENT_WEIGHT}"
echo "DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG}"
echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"

accelerate launch \
  --mixed_precision="bf16" \
  --main_process_port="${MAIN_PROCESS_PORT}" \
  --num_processes="${NUM_PROCESSES}" \
  --use_deepspeed \
  --deepspeed_config_file "${DEEPSPEED_CONFIG}" \
  --deepspeed_multinode_launcher standard \
  scripts/wan2.1_fun/train_control_customavwm.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path="${MODEL_PATH}" \
  --train_data_dir="${DATASET_DIR}" \
  --train_data_meta="${META_PATH}" \
  --image_sample_size="${IMAGE_SAMPLE_SIZE}" \
  --video_sample_size="${VIDEO_SAMPLE_SIZE}" \
  --token_sample_size="${TOKEN_SAMPLE_SIZE}" \
  --video_sample_stride="${VIDEO_SAMPLE_STRIDE}" \
  --video_sample_n_frames="${VIDEO_SAMPLE_N_FRAMES}" \
  --video_length_drop_start=0.0 \
  --video_length_drop_end=1.0 \
  --train_batch_size="${TRAIN_BATCH_SIZE}" \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers="${DATALOADER_NUM_WORKERS}" \
  --num_train_epochs="${NUM_TRAIN_EPOCHS}" \
  --checkpointing_steps="${CHECKPOINTING_STEPS}" \
  --learning_rate="${LEARNING_RATE}" \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps="${LR_WARMUP_STEPS}" \
  --seed=42 \
  --output_dir="${OUTPUT_DIR}" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm="${MAX_GRAD_NORM}" \
  --fix_sample_size "${FIX_SAMPLE_H}" "${FIX_SAMPLE_W}" \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --use_deepspeed \
  --train_mode="control_ref" \
  --control_ref_image="first_frame" \
  --add_full_ref_image_in_self_attention \
  --trainable_modules "." \
  "${EXTRA_STEP_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
