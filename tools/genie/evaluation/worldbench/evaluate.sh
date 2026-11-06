#!/usr/bin/env bash
set -euo pipefail

log_info() {
  printf '%(%H:%M:%S)T | INFO | WorldBench launcher | %s\n' -1 "$*"
}

run_stage() {
  local label="$1" log_path="$2" started=$SECONDS
  shift 2
  log_info "Starting: ${label}; log: ${log_path}"
  if "$@" 2>&1 | tee "$log_path"; then
    log_info "Completed: ${label} (elapsed $((SECONDS - started))s)"
  else
    local stage_status=$?
    log_info "Failed: ${label}; inspect ${log_path}"
    return "$stage_status"
  fi
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
gt_path="${SOURCE_INFOS:-${ROOT}/data/nus_infos_12hz/nuscenes_interp_12Hz_infos_val.pkl}"
pred_path="${SAMPLES_DIR:-${ROOT}/work_dirs/genie_inference_val150_81f}"
while (($#)); do
  case "$1" in
    --gt_path|--gt-path)
      gt_path="${2:?Missing value for $1}"
      shift 2
      ;;
    --pred_path|--pred-path)
      pred_path="${2:?Missing value for $1}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--gt_path NUSCENES_INFO_PKL] [--pred_path VIDEO_DIR]" >&2
      exit 2
      ;;
  esac
done
cd "${ROOT}"
BEVFORMER_ROOT="${BEVFORMER_ROOT:-${ROOT}/externals/W-CODA2024-Track2/BEVFormer_segmentation_detection}"
if [[ ! -f "${BEVFORMER_ROOT}/tools/test.py" ]]; then
  echo "BEVFormer sources are missing: ${BEVFORMER_ROOT}. Follow docs/inference_evaluation.md#evaluation-sources." >&2
  exit 2
fi
DATASET_ROOT="${DATASET_ROOT:-${ROOT}/work_dirs/genie_inference_val150_81f/eval/wcoda_bevformer/assets}"
SAMPLES_DIR="${pred_path}"
SOURCE_INFOS="${gt_path}"
NUSCENES_ROOT="${NUSCENES_ROOT:-${ROOT}/data/nuscenes}"
OUT_DIR="${OUT_DIR:-${ROOT}/work_dirs/genie_inference_val150_81f/eval/wcoda_bevformer}"
TIMESTAMP="${TIMESTAMP:-loworld_81f_wcodabevformer_$(date +%Y%m%d_%H%M%S)}"
PART="${PART:-nuscenes_infos_val_generated_3keyframes.pkl}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-$(awk -F, '{print NF}' <<< "${GPUS}")}"
CONDA_ENV="${CONDA_ENV:-worldbench}"

# Keep all paths anchored to the repository before entering the output directory.
BEVFORMER_ROOT="$(realpath -m "${BEVFORMER_ROOT}")"
DATASET_ROOT="$(realpath -m "${DATASET_ROOT}")"
SAMPLES_DIR="$(realpath -m "${SAMPLES_DIR}")"
SOURCE_INFOS="$(realpath -m "${SOURCE_INFOS}")"
NUSCENES_ROOT="$(realpath -m "${NUSCENES_ROOT}")"
OUT_DIR="$(realpath -m "${OUT_DIR}")"

mkdir -p "${OUT_DIR}"
RUN_LOG_DIR="${OUT_DIR}/logs/${TIMESTAMP}"
mkdir -p "${RUN_LOG_DIR}"

log_info "Starting evaluation; environment: ${CONDA_ENV}; GPUs: ${GPUS}; processes: ${NUM_PROCESSES}"
log_info "Prediction videos: ${SAMPLES_DIR}; GT annotations: ${SOURCE_INFOS}"
log_info "Output: ${OUT_DIR}; logs: ${RUN_LOG_DIR}"
run_stage "prepare generated frames and annotations" "${RUN_LOG_DIR}/prepare.log" \
conda run --no-capture-output -n "${CONDA_ENV}" python -u \
  "${ROOT}/tools/genie/evaluation/worldbench/prepare_bevformer_eval.py" \
  --samples-dir "${SAMPLES_DIR}" \
  --selection-manifest "${DATASET_ROOT}/selection_manifest.json" \
  --source-infos "${SOURCE_INFOS}" \
  --dataset-dir "${DATASET_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --overwrite \
  --nuscenes-sample-json "${NUSCENES_ROOT}/v1.0-trainval/sample.json"

CONFIG="${BEVFORMER_ROOT}/projects/configs/bevformer/bevformer_base_seg_det_150x150.py"
CHECKPOINT="${BEVFORMER_CHECKPOINT:-${ROOT}/ckpts/bevformer/bevformer_base_seg_det_150.pth}"
CHECKPOINT="$(realpath -m "${CHECKPOINT}")"
# Resolve the annotation before changing into the output directory.
ANN_FILE="$(realpath -m "${OUT_DIR}/nuscenes_infos_val_generated_3keyframes.pkl")"
LOG_ROOT="${OUT_DIR}/magicdrive-t-log/evaluation/bevformer_base_seg_det_150x150/${TIMESTAMP}/${PART}"

log_info "BEVFormer checkpoint: ${CHECKPOINT}"
cd "${OUT_DIR}"
run_stage "BEVFormer inference and metric computation" "${RUN_LOG_DIR}/evaluation.log" \
env CUDA_VISIBLE_DEVICES="${GPUS}" PYTHONUNBUFFERED=1 \
PYTHONPATH="${ROOT}/tools/genie/evaluation/worldbench/compat:${PYTHONPATH:-}" \
conda run --no-capture-output -n "${CONDA_ENV}" \
  python -u -m torch.distributed.launch \
  --use-env \
  --nproc_per_node="${NUM_PROCESSES}" \
  --master_port="${MASTER_PORT:-29503}" \
  "${ROOT}/tools/genie/evaluation/worldbench/run_bevformer_eval.py" \
  --test-script "${BEVFORMER_ROOT}/tools/test.py" \
  "${CONFIG}" "${CHECKPOINT}" \
  --launcher pytorch \
  --eval bbox \
  --cfg-options \
  "data.test.ann_file=${ANN_FILE}" \
  "data.test.data_root=${NUSCENES_ROOT}/" \
  --log_timestamp "${TIMESTAMP}" \
  --log_part "${PART}"

log_info "Collecting segmentation and detection results from ${LOG_ROOT}"
mkdir -p "${OUT_DIR}/pts_bbox"
cp "${LOG_ROOT}/segmentation_result.json" \
  "${OUT_DIR}/segmentation_result.json"
cp "${LOG_ROOT}/pts_bbox/metrics_summary.json" \
  "${OUT_DIR}/pts_bbox/metrics_summary.json"

log_info "Evaluation completed; displaying final scores; summary log: ${RUN_LOG_DIR}/summary.log"
conda run --no-capture-output -n "${CONDA_ENV}" python -u \
  "${ROOT}/tools/genie/evaluation/worldbench/summarize_metrics.py" \
  --segmentation "${OUT_DIR}/segmentation_result.json" \
  --detection "${OUT_DIR}/pts_bbox/metrics_summary.json" \
  --output "${OUT_DIR}/metrics.json" 2>&1 | tee "${RUN_LOG_DIR}/summary.log"
