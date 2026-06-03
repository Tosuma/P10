#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
STAGE1_RUNNER="${REPO_ROOT}/scripts/mami/hpfat/stage1_loop_runner.sh"
STAGE23_JOB_SCRIPT="${REPO_ROOT}/scripts/mami/hpfat/stage2_stage3_job.sh"

STAGE1_LR="4e-4"
STAGE1_MRAE="1.0"
STAGE1_NDRE="0.0"
STAGE1_NDVI="0.0"
STAGE1_DIR="hpfat/full/stage1"
STAGE1_MODEL="hpfat-andhra-re_0.1-vi_0.5"

STAGE23_DIR="hpfat/full/stage23"
STAGE23_MODEL="${STAGE1_MODEL}"

STAGE2_MODEL="${REPO_ROOT}/checkpoints/${STAGE1_DIR}/all-models/${STAGE1_MODEL}_stage1_best.pth"

mkdir -p "${REPO_ROOT}/logs/hpfat/stage23"

if [ ! -x "${STAGE1_RUNNER}" ]; then
  echo "Stage1 runner not found or not executable: ${STAGE1_RUNNER}" >&2
  exit 1
fi

if [ ! -x "${STAGE23_JOB_SCRIPT}" ]; then
  echo "Stage2+Stage3 job script not found or not executable: ${STAGE23_JOB_SCRIPT}" >&2
  exit 1
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') :: Starting Stage1 loop runner"
"${STAGE1_RUNNER}" \
  --lr "${STAGE1_LR}" \
  --loss_mrae_w "${STAGE1_MRAE}" \
  --loss_ndvi_w "${STAGE1_NDVI}" \
  --loss_ndre_w "${STAGE1_NDRE}" \
  --dir_name "${STAGE1_DIR}" \
  --model_name "${STAGE1_MODEL}"

# if [ ! -f "${STAGE2_MODEL}" ]; then
#   echo "Stage1 best model not found: ${STAGE2_MODEL}" >&2
#   exit 1
# fi

# echo "$(date '+%Y-%m-%d %H:%M:%S') :: Submitting Stage2+Stage3 job"
# job_id=$(sbatch "${STAGE23_JOB_SCRIPT}" \
#   --stage2_model "${STAGE2_MODEL}" \
#   --dir_name "${STAGE23_DIR}" \
#   --model_name "${STAGE23_MODEL}" \
#   | grep -o '[0-9]\+')

# if [ -z "${job_id}" ]; then
#   echo "$(date '+%Y-%m-%d %H:%M:%S') :: Failed to submit Stage2+Stage3 job." >&2
#   exit 1
# fi

# echo "$(date '+%Y-%m-%d %H:%M:%S') :: Stage2+Stage3 job submitted: ${job_id}"

# while squeue --me | grep -q "${job_id}"; do
#   sleep 10
# done

# echo "$(date '+%Y-%m-%d %H:%M:%S') :: Stage2+Stage3 job ${job_id} completed"
