#!/usr/bin/env bash

set -euo pipefail

MANIFEST="retry-manifest.json"
RUNS_ROOT="outputs/runs"
SUMMARY_OUTPUT="outputs/metrics/retry_combined_summary.json"
PYTHON_EXE="python"
MAX_PARALLEL=""
MAX_RETRIES=""
SBATCH_SUBMIT_RETRIES=""
POLL_SECONDS=""
SINGULARITY_IMAGE=""
VENV_ACTIVATE=""
STATUS_ROOT=""
SUMMARY_DIR=""
JOB_SCRIPT=""
DRY_RUN=0
FAIL_FAST=0

usage() {
  cat <<'EOF'
Usage:
  bash ./scripts/slurm/run_retry_manifest.sh --manifest retry-manifest.json

Options:
  --manifest PATH               Retry manifest to submit (default: retry-manifest.json)
  --runs-root PATH              Completed run root to summarize (default: outputs/runs)
  --summary-output PATH         Combined summary JSON path
  --max-parallel N              Pass through to run_masking_batch.sh
  --max-retries N               Pass through to run_masking_batch.sh
  --sbatch-submit-retries N     Pass through to run_masking_batch.sh
  --poll-seconds N              Pass through to run_masking_batch.sh
  --python PATH                 Python executable
  --singularity-image PATH      Pass through to run_masking_batch.sh
  --venv-activate PATH          Pass through to run_masking_batch.sh
  --status-root PATH            Pass through to run_masking_batch.sh
  --summary-dir PATH            Pass through to run_masking_batch.sh
  --job-script PATH             Pass through to run_masking_batch.sh
  --dry-run                     Print controller dry-run output and skip summary
  --fail-fast                   Pass through to run_masking_batch.sh
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)
      MANIFEST="$2"
      shift 2
      ;;
    --runs-root)
      RUNS_ROOT="$2"
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

if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: ${MANIFEST}" >&2
  exit 1
fi

controller_args=(--manifest "$MANIFEST" --python "$PYTHON_EXE")
if [[ -n "$MAX_PARALLEL" ]]; then controller_args+=(--max-parallel "$MAX_PARALLEL"); fi
if [[ -n "$MAX_RETRIES" ]]; then controller_args+=(--max-retries "$MAX_RETRIES"); fi
if [[ -n "$SBATCH_SUBMIT_RETRIES" ]]; then controller_args+=(--sbatch-submit-retries "$SBATCH_SUBMIT_RETRIES"); fi
if [[ -n "$POLL_SECONDS" ]]; then controller_args+=(--poll-seconds "$POLL_SECONDS"); fi
if [[ -n "$SINGULARITY_IMAGE" ]]; then controller_args+=(--singularity-image "$SINGULARITY_IMAGE"); fi
if [[ -n "$VENV_ACTIVATE" ]]; then controller_args+=(--venv-activate "$VENV_ACTIVATE"); fi
if [[ -n "$STATUS_ROOT" ]]; then controller_args+=(--status-root "$STATUS_ROOT"); fi
if [[ -n "$SUMMARY_DIR" ]]; then controller_args+=(--summary-dir "$SUMMARY_DIR"); fi
if [[ -n "$JOB_SCRIPT" ]]; then controller_args+=(--job-script "$JOB_SCRIPT"); fi
if [[ "$DRY_RUN" -eq 1 ]]; then controller_args+=(--dry-run); fi
if [[ "$FAIL_FAST" -eq 1 ]]; then controller_args+=(--fail-fast); fi

echo "Running retry manifest: ${MANIFEST}"
set +e
bash ./scripts/slurm/run_masking_batch.sh "${controller_args[@]}"
controller_exit=$?
set -e

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run complete; skipping combined summary."
  exit "$controller_exit"
fi

echo "Writing combined summary from completed runs under ${RUNS_ROOT}"
set +e
"$PYTHON_EXE" scripts/slurm/summarize_completed_runs.py --runs-root "$RUNS_ROOT" --output "$SUMMARY_OUTPUT"
summary_exit=$?
set -e

echo "controller_exit=${controller_exit}"
echo "summary_exit=${summary_exit}"
echo "summary_output=${SUMMARY_OUTPUT}"

if [[ "$summary_exit" -ne 0 ]]; then
  exit "$summary_exit"
fi
exit "$controller_exit"
