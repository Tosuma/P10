#!/bin/bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 --lr LR --loss_mrae_w MRAE --loss_ndvi_w NDVI --loss_ndre_w NDRE \
          --dir_name DIR --model_name MODEL [--max-attempts N] [--sleep-seconds N]

This runner submits stage1 training jobs repeatedly until the stage1 final model exists.
It resumes from the stage1 best model when available, or from the latest epoch checkpoint.
EOF
  exit 1
}

lr=""
mrae=""
ndvi=""
ndre=""
dir_name=""
model_name=""
max_attempts=50
sleep_seconds=60

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lr)             lr="$2"; shift 2 ;; 
    --loss_mrae_w)    mrae="$2"; shift 2 ;; 
    --loss_ndvi_w)    ndvi="$2"; shift 2 ;; 
    --loss_ndre_w)    ndre="$2"; shift 2 ;; 
    --dir_name)       dir_name="$2"; shift 2 ;; 
    --model_name)     model_name="$2"; shift 2 ;; 
    --max-attempts)   max_attempts="$2"; shift 2 ;; 
    --sleep-seconds)  sleep_seconds="$2"; shift 2 ;; 
    -h|--help)        usage ;; 
    -*)
      echo "Unknown option: $1" >&2
      usage
      ;;
    *)
      echo "Unexpected positional arg: $1" >&2
      usage
      ;;
  esac
done

: "${lr:?Missing --lr}"
: "${mrae:?Missing --loss_mrae_w}"
: "${ndvi:?Missing --loss_ndvi_w}"
: "${ndre:?Missing --loss_ndre_w}"
: "${dir_name:?Missing --dir_name}"
: "${model_name:?Missing --model_name}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TRAIN_JOB_SCRIPT="${REPO_ROOT}/scripts/mami/hpfat/stage1/train_mami_job.sh"

mkdir -p "${REPO_ROOT}/logs/hpfat"

checkpoint_dir="${REPO_ROOT}/checkpoints/${dir_name}/all-models"
final_model="${checkpoint_dir}/${model_name}_stage1_final.pth"
best_dir="${REPO_ROOT}/checkpoints/${dir_name}"
best_model="${best_dir}/${model_name}_stage1_best.pth"

attempt=0

while [ ! -f "${final_model}" ]; do
  if [ "${attempt}" -ge "${max_attempts}" ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') :: Reached max attempts (${max_attempts}) without finding ${final_model}."
    exit 1
  fi

  attempt=$((attempt + 1))
  echo "$(date '+%Y-%m-%d %H:%M:%S') :: Attempt ${attempt}. Checking for final model: ${final_model}"

  resume_model=""
  if [ -f "${best_model}" ]; then
    resume_model="${best_model}"
  else
    latest_checkpoint=$(ls -1 "${checkpoint_dir}/${model_name}_stage1_epoch_"*.pth 2>/dev/null | sort -V | tail -n 1 || true)
    if [ -n "${latest_checkpoint}" ]; then
      resume_model="${latest_checkpoint}"
    fi
  fi

  if [ -n "${resume_model}" ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') :: Resuming from checkpoint ${resume_model}"
    stage1_model_args=(--stage1_model "${resume_model}")
  else
    echo "$(date '+%Y-%m-%d %H:%M:%S') :: No resume checkpoint found; starting from scratch"
    stage1_model_args=()
  fi

  job_id=$(sbatch "${TRAIN_JOB_SCRIPT}" \
    --lr "${lr}" \
    --loss_mrae_w "${mrae}" \
    --loss_ndvi_w "${ndvi}" \
    --loss_ndre_w "${ndre}" \
    --dir_name "${dir_name}" \
    --model_name "${model_name}" \
    "${stage1_model_args[@]}" | grep -o '[0-9]\+')

  if [ -z "${job_id}" ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') :: Failed to submit job. Retrying after ${sleep_seconds}s."
    sleep "${sleep_seconds}"
    continue
  fi

  echo "$(date '+%Y-%m-%d %H:%M:%S') :: Submitted job ${job_id}"

  while squeue --me | grep -q "${job_id}"; do
    sleep 10
  done

  echo "$(date '+%Y-%m-%d %H:%M:%S') :: Job ${job_id} finished"

  if [ -f "${final_model}" ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') :: Final model created: ${final_model}"
    break
  fi

  err_file="${REPO_ROOT}/logs/hpfat/train_mami_${job_id}.err"
  if [ -f "${err_file}" ] && grep -qF "Could not lookup the current user" "${err_file}"; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') :: Retryable error detected in ${err_file}."
  fi

  echo "$(date '+%Y-%m-%d %H:%M:%S') :: Final model not found yet; restarting training loop."
  sleep "${sleep_seconds}"
done

echo "Stage1 loop runner completed. Final model: ${final_model}"
