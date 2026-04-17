#!/usr/bin/env bash

set -euo pipefail

PYTHON_EXE="python"
SKIP_EVALUATE=0
SUMMARY_OUTPUT=""
CONFIGS=()
RUN_DIRS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIGS+=("$2")
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

if ! command -v "$PYTHON_EXE" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_EXE}" >&2
  exit 1
fi

if [[ ! -d "src" || ! -d "configs" ]]; then
  echo "Run this script from tbd/masking." >&2
  exit 1
fi

export PYTHONPATH="${PWD}"

for config in "${CONFIGS[@]}"; do
  echo "Training ${config}"
  run_dir="$("$PYTHON_EXE" -m src.train --config "$config")"
  RUN_DIRS+=("$run_dir")

  if [[ "$SKIP_EVALUATE" -eq 0 ]]; then
    echo "Evaluating ${run_dir}"
    "$PYTHON_EXE" -m src.evaluate --checkpoint "${run_dir}/checkpoints/best.pt" --split test
  fi
done

if [[ ${#RUN_DIRS[@]} -gt 0 && -n "$SUMMARY_OUTPUT" ]]; then
  if [[ "$SKIP_EVALUATE" -eq 1 ]]; then
    echo "Skipping summary because --skip-evaluate was used and evaluation/test/overall_metrics.json was not generated."
  else
    echo "Writing summary to ${SUMMARY_OUTPUT}"
    "$PYTHON_EXE" -m src.summarize --runs "${RUN_DIRS[@]}" --output "$SUMMARY_OUTPUT"
  fi
fi

echo "Training run(s) completed."
