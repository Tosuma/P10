#!/usr/bin/env bash
#SBATCH --job-name=masking_task
#SBATCH --output=logs/masking/slurm/%x_%j.out
#SBATCH --error=logs/masking/slurm/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

set -euo pipefail

TASK_ID=""
GROUP=""
KIND=""
CONFIG=""
SPLIT="test"
SEED=""
ATTEMPT="1"
STATUS_DIR=""
PYTHON_EXE="python"
SINGULARITY_IMAGE="/ceph/container/pytorch/pytorch_26.02.sif"
VENV_ACTIVATE="p10_venv/bin/activate"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task-id)
      TASK_ID="$2"
      shift 2
      ;;
    --group)
      GROUP="$2"
      shift 2
      ;;
    --kind)
      KIND="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --attempt)
      ATTEMPT="$2"
      shift 2
      ;;
    --status-dir)
      STATUS_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_EXE="$2"
      shift 2
      ;;
    --singularity-image)
      SINGULARITY_IMAGE="$2"
      shift 2
      ;;
    --venv-activate)
      VENV_ACTIVATE="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

: "${TASK_ID:?Missing --task-id}"
: "${GROUP:?Missing --group}"
: "${KIND:?Missing --kind}"
: "${CONFIG:?Missing --config}"
: "${STATUS_DIR:?Missing --status-dir}"

if [[ ! -d "src" || ! -d "configs" ]]; then
  echo "Run this script from tbd/masking." >&2
  exit 1
fi

mkdir -p "$STATUS_DIR" "logs/masking/slurm"

STATUS_FILE="${STATUS_DIR}/task_${TASK_ID}_attempt_${ATTEMPT}.status"
STATUS_TMP="${STATUS_FILE}.tmp"
JOB_LOG="${STATUS_DIR}/task_${TASK_ID}_attempt_${ATTEMPT}.log"
RUN_DIR=""
EXIT_CODE=0
LAST_OUTPUT=""
STATUS_WRITTEN=0

write_status() {
  local state="$1"
  local message="$2"
  {
    printf 'task_id=%s\n' "$TASK_ID"
    printf 'group=%s\n' "$GROUP"
    printf 'kind=%s\n' "$KIND"
    printf 'config=%s\n' "$CONFIG"
    printf 'split=%s\n' "$SPLIT"
    printf 'seed=%s\n' "$SEED"
    printf 'attempt=%s\n' "$ATTEMPT"
    printf 'state=%s\n' "$state"
    printf 'exit_code=%s\n' "$EXIT_CODE"
    printf 'run_dir=%s\n' "$RUN_DIR"
    printf 'message=%s\n' "$message"
    printf 'job_log=%s\n' "$JOB_LOG"
    printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-}"
    date -u '+finished_at=%Y-%m-%dT%H:%M:%SZ'
  } > "$STATUS_TMP"
  mv "$STATUS_TMP" "$STATUS_FILE"
  STATUS_WRITTEN=1
}

on_exit() {
  EXIT_CODE=$?
  if [[ "$EXIT_CODE" -ne 0 && "$STATUS_WRITTEN" -eq 0 ]]; then
    write_status "failed" "job exited before producing a successful status"
  fi
}
trap on_exit EXIT

log_job() {
  local level="$1"
  shift
  local message="$*"
  local timestamp
  timestamp="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  printf '%s [%s] %s\n' "$timestamp" "$level" "$message" | tee -a "$JOB_LOG"
}

hostname
date
log_job "INFO" "Slurm masking job started"
log_job "INFO" "task_id=${TASK_ID} group=${GROUP} kind=${KIND} config=${CONFIG} split=${SPLIT} seed=${SEED:-none} attempt=${ATTEMPT}"
log_job "INFO" "pwd=${PWD} host=$(hostname) slurm_job_id=${SLURM_JOB_ID:-} cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
log_job "INFO" "python=${PYTHON_EXE} singularity_image=${SINGULARITY_IMAGE} venv_activate=${VENV_ACTIVATE}"
log_job "INFO" "status_file=${STATUS_FILE}"

export PYTHONPATH="${PWD}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
log_job "INFO" "PYTHONPATH=${PYTHONPATH} OMP_NUM_THREADS=${OMP_NUM_THREADS} MKL_NUM_THREADS=${MKL_NUM_THREADS}"

if command -v singularity >/dev/null 2>&1; then
  log_job "INFO" "singularity=$(command -v singularity)"
else
  log_job "ERROR" "singularity command was not found on PATH before container execution"
fi

run_inside_container() {
  local command="$1"

  singularity exec --nv "$SINGULARITY_IMAGE" \
    /bin/bash -lc "cd '$PWD' && \
      echo container_pwd=\$PWD && \
      echo container_python_before_venv=\$(command -v python || true) && \
      if [[ -f '$VENV_ACTIVATE' ]]; then \
        echo using_venv='$VENV_ACTIVATE'; \
        source '$VENV_ACTIVATE'; \
      elif [[ -f '../../$VENV_ACTIVATE' ]]; then \
        echo using_venv='../../$VENV_ACTIVATE'; \
        source '../../$VENV_ACTIVATE'; \
      elif [[ -f '../../.venv/bin/activate' ]]; then \
        echo using_venv='../../.venv/bin/activate'; \
        source '../../.venv/bin/activate'; \
      else \
        echo 'Could not find venv activation script. Tried: $VENV_ACTIVATE, ../../$VENV_ACTIVATE, ../../.venv/bin/activate' >&2; \
        exit 1; \
      fi && \
      export PYTHONPATH='$PWD' && \
      echo container_python_after_venv=\$(command -v '$PYTHON_EXE' || true) && \
      '$PYTHON_EXE' -c \"import os, sys; print('python_executable=' + sys.executable); print('python_version=' + sys.version.split()[0]); print('pythonpath=' + os.environ.get('PYTHONPATH', ''))\" && \
      ${command}"
}

run_and_capture() {
  local label="$1"
  local command="$2"
  local output
  local code

  {
    printf '\n==== %s ====\n' "$label"
    date -u '+started_at=%Y-%m-%dT%H:%M:%SZ'
    printf 'command=%s\n' "$command"
  } >> "$JOB_LOG"

  set +e
  output="$(run_inside_container "$command" 2>&1)"
  code=$?
  set -e

  printf '%s\n' "$output" | tee -a "$JOB_LOG"
  LAST_OUTPUT="$output"

  if [[ "$code" -ne 0 ]]; then
    EXIT_CODE="$code"
    write_status "failed" "${label} failed; see ${JOB_LOG}"
    exit "$code"
  fi
}

if [[ "$KIND" == "train" ]]; then
  echo "Training ${CONFIG}"
  train_command="$PYTHON_EXE -u -m src.train --config '$CONFIG'"
  if [[ -n "$SEED" ]]; then
    train_command="${train_command} --seed '$SEED'"
  fi
  run_and_capture "train" "$train_command"
  RUN_DIR="$(printf '%s\n' "$LAST_OUTPUT" | awk 'NF { last=$0 } END { print last }')"
  echo "Run directory: ${RUN_DIR}"

  echo "Evaluating ${RUN_DIR}"
  run_and_capture "evaluate" "$PYTHON_EXE -u -m src.evaluate --checkpoint '${RUN_DIR}/checkpoints/best.pt' --split '$SPLIT'"

  if [[ ! -f "${RUN_DIR}/checkpoints/best.pt" ]]; then
    echo "Expected checkpoint was not created: ${RUN_DIR}/checkpoints/best.pt" >&2
    EXIT_CODE=1
    write_status "failed" "expected checkpoint was not created; see ${JOB_LOG}"
    exit 1
  fi
  if [[ ! -f "${RUN_DIR}/evaluation/${SPLIT}/overall_metrics.json" ]]; then
    echo "Expected evaluation metrics were not created: ${RUN_DIR}/evaluation/${SPLIT}/overall_metrics.json" >&2
    EXIT_CODE=1
    write_status "failed" "expected evaluation metrics were not created; see ${JOB_LOG}"
    exit 1
  fi
elif [[ "$KIND" == "baseline" ]]; then
  echo "Evaluating unfine-tuned base model ${CONFIG} on split ${SPLIT}"
  run_and_capture "baseline" "$PYTHON_EXE -u -m src.evaluate_base --config '$CONFIG' --split '$SPLIT'"
  RUN_DIR="$(printf '%s\n' "$LAST_OUTPUT" | awk 'NF { last=$0 } END { print last }')"
  echo "Run directory: ${RUN_DIR}"

  if [[ ! -f "${RUN_DIR}/evaluation/${SPLIT}/overall_metrics.json" ]]; then
    echo "Expected evaluation metrics were not created: ${RUN_DIR}/evaluation/${SPLIT}/overall_metrics.json" >&2
    EXIT_CODE=1
    write_status "failed" "expected evaluation metrics were not created; see ${JOB_LOG}"
    exit 1
  fi
else
  echo "Unknown task kind: ${KIND}" >&2
  exit 2
fi

EXIT_CODE=0
write_status "success" "completed"
trap - EXIT
date
