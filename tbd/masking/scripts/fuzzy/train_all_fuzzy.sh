#!/usr/bin/env bash

set -euo pipefail

if [[ ! -d "src" || ! -d "configs" ]]; then
  echo "Run this script from tbd/masking." >&2
  exit 1
fi

bash ./scripts/shared/train_configs.sh \
  --config "configs/fuzzy/rgb_unetpp_resnet34.yaml" \
  --config "configs/fuzzy/rgb_unetpp_resnet50.yaml" \
  --config "configs/fuzzy/rgb_deeplabv3plus_resnet34.yaml" \
  --config "configs/fuzzy/rgb_deeplabv3plus_resnet50.yaml" \
  --config "configs/fuzzy/rgb_segformer_b0.yaml" \
  --config "configs/fuzzy/rgb_segformer_b1.yaml" \
  --config "configs/fuzzy/real_msi_unetpp_resnet34.yaml" \
  --config "configs/fuzzy/real_msi_unetpp_resnet50.yaml" \
  --config "configs/fuzzy/real_msi_deeplabv3plus_resnet34.yaml" \
  --config "configs/fuzzy/real_msi_deeplabv3plus_resnet50.yaml" \
  --config "configs/fuzzy/real_msi_segformer_b0.yaml" \
  --config "configs/fuzzy/real_msi_segformer_b1.yaml" \
  --config "configs/fuzzy/synth_msi_unetpp_resnet34.yaml" \
  --config "configs/fuzzy/synth_msi_unetpp_resnet50.yaml" \
  --config "configs/fuzzy/synth_msi_deeplabv3plus_resnet34.yaml" \
  --config "configs/fuzzy/synth_msi_deeplabv3plus_resnet50.yaml" \
  --config "configs/fuzzy/synth_msi_segformer_b0.yaml" \
  --config "configs/fuzzy/synth_msi_segformer_b1.yaml" \
  --config "configs/fuzzy/rgb_real_msi_unetpp_resnet34.yaml" \
  --config "configs/fuzzy/rgb_real_msi_unetpp_resnet50.yaml" \
  --config "configs/fuzzy/rgb_real_msi_deeplabv3plus_resnet34.yaml" \
  --config "configs/fuzzy/rgb_real_msi_deeplabv3plus_resnet50.yaml" \
  --config "configs/fuzzy/rgb_real_msi_segformer_b0.yaml" \
  --config "configs/fuzzy/rgb_real_msi_segformer_b1.yaml" \
  --config "configs/fuzzy/rgb_synth_msi_unetpp_resnet34.yaml" \
  --config "configs/fuzzy/rgb_synth_msi_unetpp_resnet50.yaml" \
  --config "configs/fuzzy/rgb_synth_msi_deeplabv3plus_resnet34.yaml" \
  --config "configs/fuzzy/rgb_synth_msi_deeplabv3plus_resnet50.yaml" \
  --config "configs/fuzzy/rgb_synth_msi_segformer_b0.yaml" \
  --config "configs/fuzzy/rgb_synth_msi_segformer_b1.yaml" \
  "$@"
