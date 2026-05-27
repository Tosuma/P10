#!/bin/bash
set -euo pipefail

bash scripts/mami/hpfat/stage3/job_runner.sh \
  --manifest scripts/mami/hpfat/stage3/manifests/tasks_manifest_tsm_2.csv \
  --status-dir logs/hpfat/stage3/status/tsm_2 \
  --task-log-dir logs/hpfat/stage3/tasks/tsm_2 \
  --singularity-image /ceph/container/pytorch/pytorch_26.01.sif \
  "$@"
