#!/usr/bin/env bash

set -euo pipefail

if [[ ! -d "src" || ! -d "configs" ]]; then
  echo "Run this script from tbd/masking." >&2
  exit 1
fi

bash ./scripts/shared/train_configs.sh \
  --config "configs/binary/synth_msi_unetpp_resnet34.yaml" \
  --config "configs/binary/synth_msi_unetpp_resnet50.yaml" \
  --config "configs/binary/synth_msi_deeplabv3plus_resnet34.yaml" \
  --config "configs/binary/synth_msi_deeplabv3plus_resnet50.yaml" \
  --config "configs/binary/synth_msi_segformer_b0.yaml" \
  --config "configs/binary/synth_msi_segformer_b1.yaml" \
  "$@"
