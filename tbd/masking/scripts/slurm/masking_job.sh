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

TASK_ID=""
GROUP=""
KIND=""
CONFIG=""
SPLIT="test"
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
RUN_DIR=""
EXIT_CODE=0

write_status() {
  local state="$1"
  local message="$2"
  {
    printf 'task_id=%s\n' "$TASK_ID"
    printf 'group=%s\n' "$GROUP"
    printf 'kind=%s\n' "$KIND"
    printf 'config=%s\n' "$CONFIG"
    printf 'split=%s\n' "$SPLIT"
    printf 'attempt=%s\n' "$ATTEMPT"
    printf 'state=%s\n' "$state"
    printf 'exit_code=%s\n' "$EXIT_CODE"
    printf 'run_dir=%s\n' "$RUN_DIR"
    printf 'message=%s\n' "$message"
    printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-}"
    date -u '+finished_at=%Y-%m-%dT%H:%M:%SZ'
  } > "$STATUS_TMP"
  mv "$STATUS_TMP" "$STATUS_FILE"
}

on_exit() {
  EXIT_CODE=$?
  if [[ "$EXIT_CODE" -ne 0 ]]; then
    write_status "failed" "job exited before producing a successful status"
  fi
}
trap on_exit EXIT

hostname
date

export PYTHONPATH="${PWD}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

run_inside_container() {
  local command="$1"

  singularity exec --nv "$SINGULARITY_IMAGE" \
    /bin/bash -lc "cd '$PWD' && source '$VENV_ACTIVATE' && export PYTHONPATH='$PWD' && ${command}"
}

if [[ "$KIND" == "train" ]]; then
  echo "Training ${CONFIG}"
  TRAIN_OUTPUT="$(run_inside_container "$PYTHON_EXE -u -m src.train --config '$CONFIG'")"
  printf '%s\n' "$TRAIN_OUTPUT"
  RUN_DIR="$(printf '%s\n' "$TRAIN_OUTPUT" | awk 'NF { last=$0 } END { print last }')"
  echo "Run directory: ${RUN_DIR}"

  echo "Evaluating ${RUN_DIR}"
  run_inside_container "$PYTHON_EXE -u -m src.evaluate --checkpoint '${RUN_DIR}/checkpoints/best.pt' --split '$SPLIT'"

  if [[ ! -f "${RUN_DIR}/checkpoints/best.pt" ]]; then
    echo "Expected checkpoint was not created: ${RUN_DIR}/checkpoints/best.pt" >&2
    exit 1
  fi
  if [[ ! -f "${RUN_DIR}/evaluation/${SPLIT}/overall_metrics.json" ]]; then
    echo "Expected evaluation metrics were not created: ${RUN_DIR}/evaluation/${SPLIT}/overall_metrics.json" >&2
    exit 1
  fi
elif [[ "$KIND" == "baseline" ]]; then
  echo "Evaluating unfine-tuned base model ${CONFIG} on split ${SPLIT}"
  BASELINE_OUTPUT="$(run_inside_container "$PYTHON_EXE -u -m src.evaluate_base --config '$CONFIG' --split '$SPLIT'")"
  printf '%s\n' "$BASELINE_OUTPUT"
  RUN_DIR="$(printf '%s\n' "$BASELINE_OUTPUT" | awk 'NF { last=$0 } END { print last }')"
  echo "Run directory: ${RUN_DIR}"

  if [[ ! -f "${RUN_DIR}/evaluation/${SPLIT}/overall_metrics.json" ]]; then
    echo "Expected evaluation metrics were not created: ${RUN_DIR}/evaluation/${SPLIT}/overall_metrics.json" >&2
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
