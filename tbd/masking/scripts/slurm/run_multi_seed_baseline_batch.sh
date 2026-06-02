#!/usr/bin/env bash

set -euo pipefail

FAMILY="binary"
SOURCE_MANIFEST=""
PASSTHROUGH_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash ./scripts/slurm/run_multi_seed_baseline_batch.sh --family binary --repeats 10 --base-seed 1000
  bash ./scripts/slurm/run_multi_seed_baseline_batch.sh --family fuzzy --repeats 10 --base-seed 1000
  bash ./scripts/slurm/run_multi_seed_baseline_batch.sh --manifest scripts/slurm/workloads/binary_baseline.json

Options:
  --family binary|fuzzy          Baseline workload family to run (default: binary)
  --manifest PATH                Baseline JSON manifest to expand instead of using --family

All other options are passed through to run_multi_seed_batch.sh, including:
  --repeats N
  --base-seed N
  --summary-output PATH
  --max-parallel N
  --max-retries N
  --sbatch-submit-retries N
  --poll-seconds N
  --python PATH
  --singularity-image PATH
  --venv-activate PATH
  --status-root PATH
  --summary-dir PATH
  --job-script PATH
  --dry-run
  --fail-fast
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --family)
      FAMILY="$2"
      shift 2
      ;;
    --manifest)
      SOURCE_MANIFEST="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! -d "src" || ! -d "configs" ]]; then
  echo "Run this script from tbd/masking." >&2
  exit 1
fi

if [[ -z "$SOURCE_MANIFEST" ]]; then
  case "$FAMILY" in
    binary)
      SOURCE_MANIFEST="scripts/slurm/workloads/binary_baseline.json"
      ;;
    fuzzy)
      SOURCE_MANIFEST="scripts/slurm/workloads/fuzzy_baseline.json"
      ;;
    *)
      echo "--family must be either 'binary' or 'fuzzy'." >&2
      usage >&2
      exit 2
      ;;
  esac
fi

bash ./scripts/slurm/run_multi_seed_batch.sh \
  --manifest "$SOURCE_MANIFEST" \
  "${PASSTHROUGH_ARGS[@]}"
