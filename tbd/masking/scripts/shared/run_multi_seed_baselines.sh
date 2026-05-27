#!/usr/bin/env bash

set -euo pipefail

PYTHON_EXE="python"
SUMMARY_OUTPUT="outputs/metrics/baseline_multi_seed_summary.json"
REPEATS=10
BASE_SEED=42
SPLIT="test"
CONFIGS=()
RUN_DIRS=()

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
    --split)
      SPLIT="$2"
      shift 2
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

if ! command -v "$PYTHON_EXE" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_EXE}" >&2
  exit 1
fi

if [[ ! -d "src" || ! -d "configs" ]]; then
  echo "Run this script from tbd/masking." >&2
  exit 1
fi

if [[ "$REPEATS" -lt 1 ]]; then
  echo "--repeats must be at least 1." >&2
  exit 1
fi

export PYTHONPATH="${PWD}"

for config in "${CONFIGS[@]}"; do
  for ((i = 1; i <= REPEATS; i++)); do
    seed=$((BASE_SEED + i - 1))
    echo "Evaluating unfine-tuned base model for ${config} run ${i}/${REPEATS} with seed ${seed} on split ${SPLIT}"
    run_dir="$("$PYTHON_EXE" -m src.evaluate_base --config "$config" --split "$SPLIT" --seed "$seed")"
    RUN_DIRS+=("$run_dir")
  done
done

if [[ ${#RUN_DIRS[@]} -gt 0 ]]; then
  if [[ "$SPLIT" != "test" ]]; then
    echo "Skipping summary because src.summarize currently reads evaluation/test/overall_metrics.json only."
  else
    echo "Writing summary to ${SUMMARY_OUTPUT}"
    "$PYTHON_EXE" -m src.summarize --runs "${RUN_DIRS[@]}" --output "$SUMMARY_OUTPUT"
  fi
fi

echo "Baseline multi-seed evaluation run(s) completed."
