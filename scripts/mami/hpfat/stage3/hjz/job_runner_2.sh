#!/bin/bash
set -euo pipefail

bash scripts/mami/hpfat/stage3/job_runner.sh \
  --manifest scripts/mami/hpfat/stage3/manifests/tasks_manifest_hjz_2.csv \
  --status-dir logs/hpfat/stage3/status/hjz_2 \
  --task-log-dir logs/hpfat/stage3/tasks/hjz_2 \
  --singularity-image /ceph/container/pytorch/pytorch_26.01.sif \
  "$@"
