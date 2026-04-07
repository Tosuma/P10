#!/usr/bin/env bash

bash ./tbd/masking/train_configs.sh \
  --config ./tbd/masking/configs/rgb_unetpp_resnet34.yaml \
  --config ./tbd/masking/configs/rgb_unetpp_resnet50.yaml \
  --config ./tbd/masking/configs/rgb_deeplabv3plus_resnet34.yaml \
  --config ./tbd/masking/configs/rgb_deeplabv3plus_resnet50.yaml \
  --config ./tbd/masking/configs/rgb_segformer_b0.yaml \
  --config ./tbd/masking/configs/rgb_segformer_b1.yaml \
  --summary-output ./tbd/masking/outputs/metrics/rgb_architectures_once.json
