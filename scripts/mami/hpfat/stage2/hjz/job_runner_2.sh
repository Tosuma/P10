#!/bin/bash
set -euo pipefail

bash scripts/mami/hpfat/stage2/job_runner.sh \
  --manifest scripts/mami/hpfat/stage2/manifests/tasks_manifest_hjz_2.csv \
  --status-dir logs/hpfat/stage2/status/hjz_2 \
  --task-log-dir logs/hpfat/stage2/tasks/hjz_2 \
  --singularity-image /ceph/container/pytorch/pytorch_26.01.sif \
  "$@"
