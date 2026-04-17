#!/usr/bin/env bash

set -euo pipefail

PYTHON_EXE="python"
DATASET_ROOT="data/weedy-rice"
SPLITS_DIR="data/splits"
GROUP_STRATEGY="datetime"
PACK_REAL_MSI=1

PATCH_CONFIGS=(
  "configs/binary/rgb.yaml"
  "configs/binary/real_msi.yaml"
  "configs/binary/synth_msi.yaml"
  "configs/binary/rgb_real_msi.yaml"
  "configs/binary/rgb_synth_msi.yaml"
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_EXE="$2"
      shift 2
      ;;
    --dataset-root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --splits-dir)
      SPLITS_DIR="$2"
      shift 2
      ;;
    --group-strategy)
      GROUP_STRATEGY="$2"
      shift 2
      ;;
    --skip-pack-real-msi)
      PACK_REAL_MSI=0
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

# if [[ "$PACK_REAL_MSI" -eq 1 ]]; then
#   echo "Packing real MSI TIFF bands into NPY files"
#   "$PYTHON_EXE" -m src.pack_real_msi \
#     --input-dir "${DATASET_ROOT}/Multispectral" \
#     --output-dir "${DATASET_ROOT}/MultispectralNPY"
# fi

echo "Creating split manifests"
"$PYTHON_EXE" -m src.create_splits \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$SPLITS_DIR" \
  --group-strategy "$GROUP_STRATEGY"

for config in "${PATCH_CONFIGS[@]}"; do
  echo "Patchifying ${config}"
  "$PYTHON_EXE" -m src.patchify --config "$config"
done

echo "Data setup completed."
