#!/usr/bin/env bash

set -euo pipefail

SOURCE_MANIFEST=""
PASSTHROUGH_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash ./scripts/slurm/run_multi_seed_baseline_batch.sh --manifest scripts/slurm/workloads/binary_baseline.json
  bash ./scripts/slurm/run_multi_seed_baseline_batch.sh --manifest scripts/slurm/workloads/binary_baseline_no_synth.json
  bash ./scripts/slurm/run_multi_seed_baseline_batch.sh --manifest scripts/slurm/workloads/binary_baseline_only_synth.json

Options:
  --manifest PATH                Baseline JSON manifest to expand into one task per seed

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
  echo "Missing --manifest." >&2
  usage >&2
  exit 2
fi

if [[ ! -f "$SOURCE_MANIFEST" ]]; then
  echo "Manifest not found: ${SOURCE_MANIFEST}" >&2
  exit 1
fi

bash ./scripts/slurm/run_multi_seed_batch.sh \
  --manifest "$SOURCE_MANIFEST" \
  "${PASSTHROUGH_ARGS[@]}"
