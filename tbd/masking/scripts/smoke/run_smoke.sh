#!/usr/bin/env bash

set -euo pipefail

PYTHON_EXE="python"
CONFIG="configs/smoke/smoke_rgb.yaml"
SUMMARY_OUTPUT="outputs/metrics/smoke_summary.json"
SKIP_EVALUATE=0
SKIP_SUMMARY=0
SKIP_PATCHIFY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_EXE="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --summary-output)
      SUMMARY_OUTPUT="$2"
      shift 2
      ;;
    --skip-patchify)
      SKIP_PATCHIFY=1
      shift
      ;;
    --skip-evaluate)
      SKIP_EVALUATE=1
      shift
      ;;
    --skip-summary)
      SKIP_SUMMARY=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if ! command -v "$PYTHON_EXE" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_EXE}" >&2
  exit 1
fi

if [[ ! -d "src" || ! -d "configs" ]]; then
  echo "Run this script from tbd/masking." >&2
  exit 1
fi

export PYTHONPATH="${PWD}"

echo "Checking smoke manifests"
for manifest in \
  "data/splits/smoke_train.csv" \
  "data/splits/smoke_val.csv" \
  "data/splits/smoke_test.csv"; do
  if [[ ! -f "$manifest" ]]; then
    echo "Missing smoke manifest: ${manifest}" >&2
    echo "Expected the checked-in smoke split files to exist before running the smoke pipeline." >&2
    exit 1
  fi
done

if [[ "$SKIP_PATCHIFY" -eq 0 ]]; then
  echo "Patchifying smoke config ${CONFIG}"
  "$PYTHON_EXE" -m src.patchify --config "$CONFIG"
fi

echo "Training smoke config ${CONFIG}"
run_dir="$("$PYTHON_EXE" -m src.train --config "$CONFIG")"
echo "Smoke run directory: ${run_dir}"

if [[ "$SKIP_EVALUATE" -eq 0 ]]; then
  echo "Evaluating ${run_dir}"
  "$PYTHON_EXE" -m src.evaluate --checkpoint "${run_dir}/checkpoints/best.pt" --split test
fi

if [[ "$SKIP_SUMMARY" -eq 0 ]]; then
  if [[ "$SKIP_EVALUATE" -eq 1 ]]; then
    echo "Skipping summary because --skip-evaluate was used and evaluation/test/overall_metrics.json was not generated."
  else
    echo "Writing smoke summary to ${SUMMARY_OUTPUT}"
    "$PYTHON_EXE" -m src.summarize --runs "$run_dir" --output "$SUMMARY_OUTPUT"
  fi
fi

echo "Smoke pipeline completed."
