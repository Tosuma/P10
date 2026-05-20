#!/bin/bash
set -euo pipefail

bash scripts/mami/hpfat/stage2/job_runner.sh \
  --manifest scripts/mami/hpfat/stage2/manifests/tasks_manifest_mhj_1.csv \
  --status-dir logs/hpfat/stage2/status/mhj_1 \
  --task-log-dir logs/hpfat/stage2/tasks/mhj_1 \
  "$@"
