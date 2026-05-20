#!/bin/bash
set -euo pipefail

bash scripts/mami/hpfat/stage2/job_runner.sh \
  --manifest scripts/mami/hpfat/stage2/manifests/tasks_manifest_hjz_1.csv \
  --status-dir logs/hpfat/stage2/status/hjz_1 \
  --task-log-dir logs/hpfat/stage2/tasks/hjz_1 \
  "$@"
