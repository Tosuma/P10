#!/usr/bin/env bash

set -euo pipefail

DEFAULT_REPEATS=10
DEFAULT_BASE_SEED=42
DEFAULT_MAX_PARALLEL=8
PYTHON_EXE="python"
SINGULARITY_IMAGE="/ceph/container/pytorch/pytorch_26.02.sif"
VENV_ACTIVATE="p10_venv/bin/activate"
STATUS_ROOT="outputs/slurm/status"
SUMMARY_DIR="outputs/metrics"
MANIFEST_OUTPUT_ROOT="outputs/slurm/manifests"
JOB_SCRIPT="scripts/slurm/masking_job.sh"
REPEATS="$DEFAULT_REPEATS"
BASE_SEED="$DEFAULT_BASE_SEED"
MAX_PARALLEL="$DEFAULT_MAX_PARALLEL"
MAX_RETRIES=""
SBATCH_SUBMIT_RETRIES=""
POLL_SECONDS=""
SOURCE_MANIFEST=""
SUMMARY_OUTPUT=""
DRY_RUN=0
FAIL_FAST=0
CONFIGS=()

usage() {
  cat <<'EOF'
Usage:
  bash ./scripts/slurm/run_multi_seed_batch.sh --config configs/binary/rgb.yaml
  bash ./scripts/slurm/run_multi_seed_batch.sh --manifest scripts/slurm/workloads/binary_train.json
  bash ./scripts/slurm/run_multi_seed_batch.sh --manifest scripts/slurm/workloads/binary_baseline.json

Options:
  --config PATH                  Training config to repeat; may be passed multiple times
  --manifest PATH                JSON train or baseline manifest to expand into one task per seed
  --repeats N                    Number of runs per config (default: 10)
  --base-seed N                  First seed to use (default: 42)
  --summary-output PATH          Final combined summary JSON path
  --max-parallel N               Active Slurm jobs to keep running (default: 16)
  --max-retries N                Retries per task after validation failure
  --sbatch-submit-retries N      Retries when sbatch does not return a job id
  --poll-seconds N               Seconds between squeue polls
  --python PATH                  Python executable inside the runtime environment
  --singularity-image PATH       Singularity image used by each Slurm job
  --venv-activate PATH           Venv activation script inside the container
  --status-root PATH             Root directory for task status files
  --summary-dir PATH             Directory for summary JSON files
  --job-script PATH              Slurm job script to submit
  --dry-run                      Expand and print tasks without submitting jobs
  --fail-fast                    Stop submitting new work after the first permanent failure
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIGS+=("$2")
      shift 2
      ;;
    --manifest)
      SOURCE_MANIFEST="$2"
      shift 2
      ;;
    --repeats)
      REPEATS="$2"
      shift 2
      ;;
    --base-seed)
      BASE_SEED="$2"
      shift 2
      ;;
    --summary-output)
      SUMMARY_OUTPUT="$2"
      shift 2
      ;;
    --max-parallel)
      MAX_PARALLEL="$2"
      shift 2
      ;;
    --max-retries)
      MAX_RETRIES="$2"
      shift 2
      ;;
    --sbatch-submit-retries)
      SBATCH_SUBMIT_RETRIES="$2"
      shift 2
      ;;
    --poll-seconds)
      POLL_SECONDS="$2"
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
    --status-root)
      STATUS_ROOT="$2"
      shift 2
      ;;
    --summary-dir)
      SUMMARY_DIR="$2"
      shift 2
      ;;
    --job-script)
      JOB_SCRIPT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --fail-fast)
      FAIL_FAST=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "src" || ! -d "configs" ]]; then
  echo "Run this script from tbd/masking." >&2
  exit 1
fi

if [[ -z "$SOURCE_MANIFEST" && ${#CONFIGS[@]} -eq 0 ]]; then
  echo "At least one --config or one --manifest is required." >&2
  exit 2
fi

if [[ "$REPEATS" -lt 1 ]]; then
  echo "--repeats must be at least 1." >&2
  exit 2
fi

if [[ "$MAX_PARALLEL" -lt 1 ]]; then
  echo "--max-parallel must be at least 1." >&2
  exit 2
fi

if [[ -n "$SOURCE_MANIFEST" && ! -f "$SOURCE_MANIFEST" ]]; then
  echo "Manifest not found: ${SOURCE_MANIFEST}" >&2
  exit 1
fi

if [[ ! -f "$JOB_SCRIPT" ]]; then
  echo "Slurm job script not found: ${JOB_SCRIPT}" >&2
  exit 1
fi

if ! command -v "$PYTHON_EXE" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_EXE}" >&2
  exit 1
fi

mkdir -p "$STATUS_ROOT" "$SUMMARY_DIR" "$MANIFEST_OUTPUT_ROOT"

timestamp_utc() {
  date -u '+%Y%m%dT%H%M%SZ'
}

RUN_ID="$(timestamp_utc)"
if [[ -n "$SOURCE_MANIFEST" ]]; then
  BASE_NAME="$(basename "$SOURCE_MANIFEST")"
  BASE_NAME="${BASE_NAME%.*}"
else
  BASE_NAME="configs"
fi
BATCH_NAME="multi_seed_${BASE_NAME}"
STATUS_DIR="${STATUS_ROOT}/${BATCH_NAME}_${RUN_ID}"
EXPANDED_MANIFEST="${MANIFEST_OUTPUT_ROOT}/${BATCH_NAME}_${RUN_ID}.json"
FAILED_REPORT="${STATUS_DIR}/failed_tasks.tsv"

if [[ -z "$SUMMARY_OUTPUT" ]]; then
  SUMMARY_OUTPUT="${SUMMARY_DIR}/slurm_${BATCH_NAME}_${RUN_ID}_summary.json"
fi

manifest_args=(
  --repeats "$REPEATS"
  --base-seed "$BASE_SEED"
  --group-name "$BATCH_NAME"
  --output "$EXPANDED_MANIFEST"
)
if [[ -n "$SOURCE_MANIFEST" ]]; then
  manifest_args+=(--manifest "$SOURCE_MANIFEST")
fi
for config in "${CONFIGS[@]}"; do
  manifest_args+=(--config "$config")
done

"$PYTHON_EXE" scripts/slurm/write_multi_seed_manifest.py "${manifest_args[@]}"

echo "Expanded manifest: ${EXPANDED_MANIFEST}"
echo "Status directory: ${STATUS_DIR}"

controller_args=(
  --manifest "$EXPANDED_MANIFEST"
  --max-parallel "$MAX_PARALLEL"
  --python "$PYTHON_EXE"
  --singularity-image "$SINGULARITY_IMAGE"
  --venv-activate "$VENV_ACTIVATE"
  --status-root "$STATUS_ROOT"
  --summary-dir "$SUMMARY_DIR"
  --job-script "$JOB_SCRIPT"
  --status-dir "$STATUS_DIR"
)
if [[ -n "$MAX_RETRIES" ]]; then
  controller_args+=(--max-retries "$MAX_RETRIES")
fi
if [[ -n "$SBATCH_SUBMIT_RETRIES" ]]; then
  controller_args+=(--sbatch-submit-retries "$SBATCH_SUBMIT_RETRIES")
fi
if [[ -n "$POLL_SECONDS" ]]; then
  controller_args+=(--poll-seconds "$POLL_SECONDS")
fi
if [[ "$DRY_RUN" -eq 1 ]]; then
  controller_args+=(--dry-run)
fi
if [[ "$FAIL_FAST" -eq 1 ]]; then
  controller_args+=(--fail-fast)
fi

set +e
bash ./scripts/slurm/run_masking_batch.sh "${controller_args[@]}"
controller_exit=$?
set -e

if [[ "$controller_exit" -ne 0 ]]; then
  echo "Slurm batch controller exited with code ${controller_exit}." >&2
  echo "Status directory: ${STATUS_DIR}" >&2
  if [[ -f "$FAILED_REPORT" ]]; then
    echo "Failed task report: ${FAILED_REPORT}" >&2
  fi
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run complete."
  exit 0
fi

mapfile -t RUN_DIRS < <("$PYTHON_EXE" scripts/slurm/collect_successful_runs.py --status-dir "$STATUS_DIR")

if [[ "${#RUN_DIRS[@]}" -gt 0 ]]; then
  echo "Writing combined multi-seed summary to ${SUMMARY_OUTPUT}"
  "$PYTHON_EXE" -m src.summarize --runs "${RUN_DIRS[@]}" --output "$SUMMARY_OUTPUT"
else
  echo "No successful runs were found under ${STATUS_DIR}; skipping combined summary." >&2
fi

exit "$controller_exit"
