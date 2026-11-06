#!/usr/bin/env bash
set -euo pipefail
log_info() {
  printf '%(%H:%M:%S)T | INFO | launcher | %s\n' -1 "$*"
}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"
gt_path="${NUSCENES_ROOT:-data/nuscenes}"
pred_path="${WCODA_PRED_DIR:-work_dirs/genie_inference_val150_81f}"
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
      echo "Usage: $0 [--gt_path NUSCENES_ROOT] [--pred_path VIDEO_DIR]" >&2
      exit 2
      ;;
  esac
done
dataset="${WCODA_DATASET_DIR:-work_dirs/genie_inference_val150_81f/eval/wcoda_fp64_batch50/assets}"
predictions="${pred_path}"
out="${WCODA_OUT_DIR:-work_dirs/genie_inference_val150_81f/eval/wcoda_fp64_batch50}"
meta="${WCODA_META_PATH:-${dataset}/genie_dataset/meta.json}"
nuscenes="${gt_path}"
i3d_weights="${I3D_WEIGHTS_DIR:-ckpts/fvd}"
batch_size=50
conda_env="${CONDA_ENV:-loworld}"
wcoda_root="${WCODA_ROOT:-externals/W-CODA2024-Track2}"
if [[ ! -f "${wcoda_root}/fvd/videogpt/fvd.py" ]]; then
  echo "W-CODA sources are missing: ${wcoda_root}. Follow docs/inference_evaluation.md#evaluation-sources." >&2
  exit 2
fi
gpu_ids="${WCODA_GPU_IDS:-0,1,2,3,4,5,6,7}"
IFS=, read -r -a gpus <<< "$gpu_ids"
if [[ "${#gpus[@]}" -ne 8 ]]; then
  echo "WCODA_GPU_IDS must list exactly eight physical GPU IDs, got: $gpu_ids" >&2
  exit 2
fi
mkdir -p "$out"
common=(--out-dir "$out" --dataset-dir "$dataset" --pred-dir "$predictions"
        --meta-path "$meta" --nuscenes-root "$nuscenes"
        --i3d-weights-dir "$i3d_weights" --wcoda-root "$wcoda_root"
        --batch-size "$batch_size")
log_info "Starting W-CODA FVD evaluation: 81f and 37f, 150 clips, six cameras"
log_info "Environment: ${conda_env}; GPUs: ${gpu_ids}; output: ${out}"
log_info "Validating inputs; log: ${out}/prepare.log"
conda run --no-capture-output -n "$conda_env" python -u tools/genie/evaluation/wcoda_fvd.py \
  "${common[@]}" --prepare 2>&1 | tee "$out/prepare.log"
view_groups=(
  "CAM_FRONT_LEFT CAM_FRONT"
  "CAM_FRONT_RIGHT CAM_BACK_RIGHT"
  "CAM_BACK"
  "CAM_BACK_LEFT"
)
pids=()
job_logs=()
gpu_index=0
run_extraction() {
  local frames="$1" gpu="$2" log_path="$3"
  shift 3
  log_info "Starting ${frames}f extraction on GPU ${gpu}: $*; log: ${log_path}"
  if env CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
    conda run --no-capture-output -n "$conda_env" python -u tools/genie/evaluation/wcoda_fvd.py \
    "${common[@]}" --frames "$frames" --views "$@" \
    2>&1 | tee "$log_path"; then
    log_info "Completed ${frames}f extraction on GPU ${gpu}"
  else
    log_info "Failed ${frames}f extraction on GPU ${gpu}; log: ${log_path}"
    return 1
  fi
}
log_info "Launching eight feature extraction workers; progress is printed below"
for frames in 81 37; do
  for view_group in "${view_groups[@]}"; do
    read -r -a views <<< "$view_group"
    job_log="$out/extract_${frames}f_part${gpu_index}.log"
    run_extraction "$frames" "${gpus[$gpu_index]}" "$job_log" "${views[@]}" &
    pids+=("$!")
    job_logs+=("$job_log")
    gpu_index=$((gpu_index + 1))
  done
done
status=0
for job_index in "${!pids[@]}"; do
  if ! wait "${pids[$job_index]}"; then
    log_info "Worker failed; log: ${job_logs[$job_index]}"
    status=1
  fi
done
if [[ "$status" != 0 ]]; then
  echo "A feature extraction job failed; inspect extraction logs in $out" >&2
  exit 1
fi
for frames in 81 37; do
  log_info "Aggregating ${frames}f features and computing FVD; log: ${out}/fvd_${frames}f.log"
  conda run --no-capture-output -n "$conda_env" python -u tools/genie/evaluation/wcoda_fvd.py \
    "${common[@]}" --frames "$frames" \
    --aggregate 2>&1 | tee "$out/fvd_${frames}f.log"
done
log_info "Evaluation completed; results: ${out}/fvd_81f.json and ${out}/fvd_37f.json"
conda run --no-capture-output -n "$conda_env" python -u tools/genie/evaluation/summarize_fvd.py \
  --out-dir "$out" 2>&1 | tee "$out/summary.log"
