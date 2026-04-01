#!/usr/bin/env bash

set -euo pipefail

PYTHON_EXE="python"
SUMMARY_OUTPUT="./tbd/masking/outputs/metrics/multi_seed_summary.json"
REPEATS=3
SKIP_EVALUATE=0
BASE_SEED=12345
CONFIGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIGS+=("$2")
      shift 2
      ;;
    --repeats)
      REPEATS="$2"
      shift 2
      ;;
    --python)
      PYTHON_EXE="$2"
      shift 2
      ;;
    --summary-output)
      SUMMARY_OUTPUT="$2"
      shift 2
      ;;
    --base-seed)
      BASE_SEED="$2"
      shift 2
      ;;
    --skip-evaluate)
      SKIP_EVALUATE=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  echo "At least one --config is required." >&2
  exit 1
fi

export PYTHONPATH="./tbd/masking"

if ! command -v "$PYTHON_EXE" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_EXE}" >&2
  exit 1
fi

RUN_DIRS=()

for config in "${CONFIGS[@]}"; do
  for ((i = 1; i <= REPEATS; i++)); do
    seed=$((BASE_SEED + i - 1))
    echo "Training ${config} run ${i}/${REPEATS} with seed ${seed}"

    run_dir="$("$PYTHON_EXE" -m src.train --config "$config" --seed "$seed")"
    RUN_DIRS+=("$run_dir")

    if [[ "$SKIP_EVALUATE" -eq 0 ]]; then
      echo "Evaluating ${run_dir}"
      "$PYTHON_EXE" -m src.evaluate --checkpoint "${run_dir}/checkpoints/best.pt" --split test
    fi
  done
done

if [[ ${#RUN_DIRS[@]} -gt 0 ]]; then
  if [[ "$SKIP_EVALUATE" -eq 1 ]]; then
    echo "Skipping summary because --skip-evaluate was used and evaluation metrics were not generated."
  else
    echo "Writing summaries to ${SUMMARY_OUTPUT}"
    "$PYTHON_EXE" -m src.summarize --runs "${RUN_DIRS[@]}" --output "$SUMMARY_OUTPUT"
  fi
fi
