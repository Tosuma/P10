#!/usr/bin/env bash

# Edit these defaults to match the cluster. The job script also contains the
# SBATCH resource defaults so it can be submitted directly for debugging.
MAX_PARALLEL=8
# Number of times a task is retried after a submitted job finishes without
# valid outputs. Set this to 10, 50, etc. if the cluster is flaky.
MAX_RETRIES=3
# Number of times to retry sbatch itself when Slurm does not return a job id.
SBATCH_SUBMIT_RETRIES=3
POLL_SECONDS=10
PYTHON_EXE="python"
SINGULARITY_IMAGE="/ceph/container/pytorch/pytorch_26.02.sif"
VENV_ACTIVATE="p10_venv/bin/activate"
LOG_DIR="logs/masking/slurm"
STATUS_ROOT="outputs/slurm/status"
SUMMARY_DIR="outputs/metrics"
JOB_SCRIPT="scripts/slurm/masking_job.sh"
DRY_RUN=0
FAIL_FAST=0
MANIFEST=""
STATUS_DIR_OVERRIDE=""

usage() {
  cat <<'EOF'
Usage:
  bash ./scripts/slurm/run_masking_batch.sh --manifest scripts/slurm/workloads/all.json

Options:
  --manifest PATH             JSON manifest with a top-level tasks list
  --max-parallel N            Number of active Slurm jobs to keep running (default: 6)
  --max-retries N             Retries per task after validation failure (default: 3)
  --sbatch-submit-retries N   Retries when sbatch does not return a job id (default: 3)
  --poll-seconds N            Seconds between squeue polls (default: 10)
  --python PATH               Python executable inside the runtime environment
  --singularity-image PATH    Singularity image used by each Slurm job
  --venv-activate PATH        Venv activation script inside the container
  --status-root PATH          Root directory for task status files
  --summary-dir PATH          Directory for final summary JSON files
  --job-script PATH           Slurm job script to submit
  --status-dir PATH           Explicit status directory for this batch
  --dry-run                   Parse and print tasks without submitting jobs
  --fail-fast                 Stop submitting new work after the first permanent failure
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)
      MANIFEST="$2"
      shift 2
      ;;
    --max-parallel)
      MAX_PARALLEL="$2"
      shift 2
      ;;
    --max-retries)
      MAX_RETRIES="$2"
      shift 2
      ;;
    --sbatch-submit-retries)
      SBATCH_SUBMIT_RETRIES="$2"
      shift 2
      ;;
    --poll-seconds)
      POLL_SECONDS="$2"
      shift 2
      ;;
    --python)
      PYTHON_EXE="$2"
      shift 2
      ;;
    --singularity-image)
      SINGULARITY_IMAGE="$2"
      shift 2
      ;;
    --venv-activate)
      VENV_ACTIVATE="$2"
      shift 2
      ;;
    --status-root)
      STATUS_ROOT="$2"
      shift 2
      ;;
    --summary-dir)
      SUMMARY_DIR="$2"
      shift 2
      ;;
    --job-script)
      JOB_SCRIPT="$2"
      shift 2
      ;;
    --status-dir)
      STATUS_DIR_OVERRIDE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --fail-fast)
      FAIL_FAST=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$MANIFEST" ]]; then
  echo "Missing --manifest." >&2
  usage >&2
  exit 2
fi

if [[ ! -d "src" || ! -d "configs" ]]; then
  echo "Run this script from tbd/masking." >&2
  exit 1
fi

if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: ${MANIFEST}" >&2
  exit 1
fi

if [[ ! -f "$JOB_SCRIPT" ]]; then
  echo "Slurm job script not found: ${JOB_SCRIPT}" >&2
  exit 1
fi

if [[ "$MAX_PARALLEL" -lt 1 ]]; then
  echo "--max-parallel must be at least 1." >&2
  exit 2
fi

if [[ "$MAX_RETRIES" -lt 0 ]]; then
  echo "--max-retries must be at least 0." >&2
  exit 2
fi

if [[ "$SBATCH_SUBMIT_RETRIES" -lt 1 ]]; then
  echo "--sbatch-submit-retries must be at least 1." >&2
  exit 2
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
  if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch was not found on PATH." >&2
    exit 1
  fi
  if ! command -v squeue >/dev/null 2>&1; then
    echo "squeue was not found on PATH." >&2
    exit 1
  fi
fi

MANIFEST_NAME="$(basename "$MANIFEST")"
MANIFEST_NAME="${MANIFEST_NAME%.*}"
RUN_ID="$(date -u '+%Y%m%dT%H%M%SZ')"
STATUS_DIR="${STATUS_ROOT}/${MANIFEST_NAME}_${RUN_ID}"
if [[ -n "$STATUS_DIR_OVERRIDE" ]]; then
  STATUS_DIR="$STATUS_DIR_OVERRIDE"
fi
FAILED_REPORT="${STATUS_DIR}/failed_tasks.tsv"
CONTROLLER_LOG="${STATUS_DIR}/controller.log"
LOCK_DIR="${STATUS_ROOT}/.masking_batch_controller.lock"

log() {
  local level="$1"
  shift
  local message="$*"
  local timestamp
  timestamp="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  printf '%s [%s] %s\n' "$timestamp" "$level" "$message" | tee -a "$CONTROLLER_LOG"
}

log_error() {
  local message="$*"
  log "ERROR" "$message" >&2
}

cleanup_lock() {
  if [[ -n "${LOCK_DIR:-}" && -d "${LOCK_DIR:-}" ]]; then
    rm -f "${LOCK_DIR}/owner" 2>/dev/null || true
    rmdir "$LOCK_DIR" 2>/dev/null || true
  fi
}

mkdir -p "$LOG_DIR" "$STATUS_ROOT" "$SUMMARY_DIR"
mkdir -p "$STATUS_DIR"
log "INFO" "Controller started"
log "INFO" "manifest=${MANIFEST} max_parallel=${MAX_PARALLEL} max_retries=${MAX_RETRIES} sbatch_submit_retries=${SBATCH_SUBMIT_RETRIES} poll_seconds=${POLL_SECONDS}"
log "INFO" "python=${PYTHON_EXE} singularity_image=${SINGULARITY_IMAGE} venv_activate=${VENV_ACTIVATE}"
log "INFO" "status_dir=${STATUS_DIR} summary_dir=${SUMMARY_DIR} log_dir=${LOG_DIR}"

if [[ "$DRY_RUN" -eq 0 ]]; then
  if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    log_error "Another masking Slurm batch controller appears to be running: ${LOCK_DIR}"
    log_error "Remove the lock only after confirming no controller is active."
    exit 1
  fi
  {
    printf 'pid=%s\n' "$$"
    printf 'manifest=%s\n' "$MANIFEST"
    date -u '+started_at=%Y-%m-%dT%H:%M:%SZ'
  } > "${LOCK_DIR}/owner"
  trap cleanup_lock EXIT
  log "INFO" "Acquired controller lock ${LOCK_DIR}"
fi

TASK_GROUPS=()
TASK_KINDS=()
TASK_CONFIGS=()
TASK_SPLITS=()
TASK_SEEDS=()
TASK_RESUME_CHECKPOINTS=()
TASK_ATTEMPTS=()
TASK_STATES=()
TASK_RUN_DIRS=()
TASK_JOB_IDS=()
TASK_COUNT=0
EMPTY_FIELD="__MASKING_EMPTY__"

while IFS=$'\t' read -r group kind config split seed resume_checkpoint extra; do
  group="${group%$'\r'}"
  kind="${kind%$'\r'}"
  config="${config%$'\r'}"
  split="${split%$'\r'}"
  seed="${seed%$'\r'}"
  resume_checkpoint="${resume_checkpoint%$'\r'}"
  if [[ "$seed" == "$EMPTY_FIELD" ]]; then
    seed=""
  fi
  if [[ "$resume_checkpoint" == "$EMPTY_FIELD" ]]; then
    resume_checkpoint=""
  fi
  if [[ "$kind" != "train" && "$kind" != "baseline" ]]; then
    echo "Unsupported task kind '${kind}' in ${MANIFEST}." >&2
    exit 2
  fi
  if [[ -n "$resume_checkpoint" && "$kind" != "train" ]]; then
    echo "Task ${TASK_COUNT} has resume_checkpoint but kind is '${kind}', expected 'train'." >&2
    exit 2
  fi
  if [[ -z "$split" ]]; then
    split="test"
  fi
  TASK_GROUPS+=("$group")
  TASK_KINDS+=("$kind")
  TASK_CONFIGS+=("$config")
  TASK_SPLITS+=("$split")
  TASK_SEEDS+=("$seed")
  TASK_RESUME_CHECKPOINTS+=("$resume_checkpoint")
  TASK_ATTEMPTS+=(0)
  TASK_STATES+=("pending")
  TASK_RUN_DIRS+=("")
  TASK_JOB_IDS+=("")
  TASK_COUNT=$((TASK_COUNT + 1))
done < <("$PYTHON_EXE" scripts/slurm/read_manifest.py "$MANIFEST")

if [[ "$TASK_COUNT" -eq 0 ]]; then
  echo "Manifest contains no tasks: ${MANIFEST}" >&2
  exit 1
fi

log "INFO" "Loaded ${TASK_COUNT} task(s) from ${MANIFEST}"
log "INFO" "Status directory: ${STATUS_DIR}"

status_value() {
  local file="$1"
  local key="$2"
  awk -F= -v wanted="$key" '$1 == wanted { sub("^[^=]*=", ""); print; exit }' "$file"
}

expected_output_exists() {
  local task_index="$1"
  local run_dir="$2"
  local kind="${TASK_KINDS[$task_index]}"
  local split="${TASK_SPLITS[$task_index]}"

  if [[ -z "$run_dir" || ! -d "$run_dir" ]]; then
    return 1
  fi
  if [[ "$kind" == "train" && ! -f "${run_dir}/checkpoints/best.pt" ]]; then
    return 1
  fi
  [[ -f "${run_dir}/evaluation/${split}/overall_metrics.json" ]]
}

validate_task() {
  local task_index="$1"
  local job_id="$2"
  local attempt="${TASK_ATTEMPTS[$task_index]}"
  local status_file="${STATUS_DIR}/task_${task_index}_attempt_${attempt}.status"
  local out_file="${LOG_DIR}/masking_task_${job_id}.out"
  local err_file="${LOG_DIR}/masking_task_${job_id}.err"

  if [[ ! -f "$out_file" ]]; then
    log_error "Task ${task_index}: missing Slurm stdout log ${out_file}"
    return 1
  fi
  if [[ ! -f "$err_file" ]]; then
    log_error "Task ${task_index}: missing Slurm stderr log ${err_file}"
    return 1
  fi
  if grep -q "Could not lookup the current user" "$err_file"; then
    log_error "Task ${task_index}: transient user lookup failure in ${err_file}"
    return 1
  fi
  if [[ ! -f "$status_file" ]]; then
    log_error "Task ${task_index}: missing status file ${status_file}"
    return 1
  fi

  local state
  state="$(status_value "$status_file" "state")"
  if [[ "$state" != "success" ]]; then
    local message
    local job_log
    local seed
    message="$(status_value "$status_file" "message")"
    job_log="$(status_value "$status_file" "job_log")"
    seed="$(status_value "$status_file" "seed")"
    log_error "Task ${task_index}: status file reports state=${state}; seed=${seed}; message=${message}; job_log=${job_log}"
    return 1
  fi

  local run_dir
  run_dir="$(status_value "$status_file" "run_dir")"
  if ! expected_output_exists "$task_index" "$run_dir"; then
    log_error "Task ${task_index}: expected run outputs were not found under ${run_dir}"
    return 1
  fi

  TASK_RUN_DIRS[$task_index]="$run_dir"
  return 0
}

submit_task() {
  local task_index="$1"
  local attempt="${TASK_ATTEMPTS[$task_index]}"
  local group="${TASK_GROUPS[$task_index]}"
  local kind="${TASK_KINDS[$task_index]}"
  local config="${TASK_CONFIGS[$task_index]}"
  local split="${TASK_SPLITS[$task_index]}"
  local seed="${TASK_SEEDS[$task_index]}"
  local resume_checkpoint="${TASK_RESUME_CHECKPOINTS[$task_index]}"
  local output
  local job_id
  local submit_try

  for ((submit_try = 1; submit_try <= SBATCH_SUBMIT_RETRIES; submit_try++)); do
    log "INFO" "Submitting task ${task_index} try ${submit_try}/${SBATCH_SUBMIT_RETRIES}: group=${group} kind=${kind} config=${config} split=${split} seed=${seed:-none} resume_checkpoint=${resume_checkpoint:-none} attempt=${attempt}"
    output="$(sbatch "$JOB_SCRIPT" \
      --task-id "$task_index" \
      --group "$group" \
      --kind "$kind" \
      --config "$config" \
      --split "$split" \
      --seed "$seed" \
      --resume-checkpoint "$resume_checkpoint" \
      --attempt "$attempt" \
      --status-dir "$STATUS_DIR" \
      --python "$PYTHON_EXE" \
      --singularity-image "$SINGULARITY_IMAGE" \
      --venv-activate "$VENV_ACTIVATE" 2>&1 || true)"

    job_id="$(printf '%s\n' "$output" | awk '/Submitted batch job/ { print $4; exit }')"
    if [[ -n "$job_id" ]]; then
      TASK_STATES[$task_index]="running"
      TASK_JOB_IDS[$task_index]="$job_id"
      log "INFO" "Submitted task ${task_index} (${group}, ${kind}, ${config}, seed=${seed:-none}, resume_checkpoint=${resume_checkpoint:-none}) as job ${job_id}, attempt ${attempt}"
      return 0
    fi

    log_error "Task ${task_index}: sbatch did not return a job id on submit try ${submit_try}: ${output}"
    sleep "$POLL_SECONDS"
  done

  return 1
}

running_job_exists() {
  local job_id="$1"
  squeue --me -h -j "$job_id" 2>/dev/null | grep -q "$job_id"
}

active_count() {
  local count=0
  local i
  for ((i = 0; i < TASK_COUNT; i++)); do
    if [[ "${TASK_STATES[$i]}" == "running" ]]; then
      count=$((count + 1))
    fi
  done
  printf '%s\n' "$count"
}

has_pending_work() {
  local i
  for ((i = 0; i < TASK_COUNT; i++)); do
    if [[ "${TASK_STATES[$i]}" == "pending" ]]; then
      return 0
    fi
  done
  return 1
}

next_pending_task() {
  local i
  for ((i = 0; i < TASK_COUNT; i++)); do
    if [[ "${TASK_STATES[$i]}" == "pending" ]]; then
      printf '%s\n' "$i"
      return 0
    fi
  done
  return 1
}

if [[ "$DRY_RUN" -eq 1 ]]; then
  for ((i = 0; i < TASK_COUNT; i++)); do
    printf 'task=%s group=%s kind=%s config=%s split=%s seed=%s resume_checkpoint=%s\n' \
      "$i" "${TASK_GROUPS[$i]}" "${TASK_KINDS[$i]}" "${TASK_CONFIGS[$i]}" "${TASK_SPLITS[$i]}" "${TASK_SEEDS[$i]:-}" "${TASK_RESUME_CHECKPOINTS[$i]:-}"
  done
  log "INFO" "Dry run complete. No Slurm jobs were submitted."
  exit 0
fi

permanent_failures=0
stop_submissions=0

while has_pending_work || [[ "$(active_count)" -gt 0 ]]; do
  while [[ "$stop_submissions" -eq 0 ]] && has_pending_work && [[ "$(active_count)" -lt "$MAX_PARALLEL" ]]; do
    task_index="$(next_pending_task)"
    TASK_ATTEMPTS[$task_index]=$((TASK_ATTEMPTS[$task_index] + 1))
    if ! submit_task "$task_index"; then
      if [[ "${TASK_ATTEMPTS[$task_index]}" -le "$MAX_RETRIES" ]]; then
        log_error "Task ${task_index}: will retry after sbatch submission failure"
        TASK_STATES[$task_index]="pending"
      else
        log_error "Task ${task_index}: permanent failure after sbatch submission failures"
        TASK_STATES[$task_index]="failed"
        permanent_failures=$((permanent_failures + 1))
        if [[ "$FAIL_FAST" -eq 1 ]]; then
          stop_submissions=1
        fi
      fi
    fi
  done

  for ((i = 0; i < TASK_COUNT; i++)); do
    if [[ "${TASK_STATES[$i]}" != "running" ]]; then
      continue
    fi

    job_id="${TASK_JOB_IDS[$i]}"
    if running_job_exists "$job_id"; then
      continue
    fi

    log "INFO" "Job ${job_id} for task ${i} has left the Slurm queue; validating output"
    if validate_task "$i" "$job_id"; then
      TASK_STATES[$i]="done"
      log "INFO" "Task ${i} completed successfully"
    else
      if [[ "${TASK_ATTEMPTS[$i]}" -le "$MAX_RETRIES" ]]; then
        log_error "Task ${i}: validation failed; re-queueing for retry"
        TASK_STATES[$i]="pending"
        TASK_JOB_IDS[$i]=""
      else
        log_error "Task ${i}: permanent failure after ${TASK_ATTEMPTS[$i]} attempt(s)"
        TASK_STATES[$i]="failed"
        TASK_JOB_IDS[$i]=""
        permanent_failures=$((permanent_failures + 1))
        if [[ "$FAIL_FAST" -eq 1 ]]; then
          stop_submissions=1
        fi
      fi
    fi
  done

  if [[ "$stop_submissions" -eq 1 && "$(active_count)" -eq 0 ]]; then
    break
  fi

  if has_pending_work || [[ "$(active_count)" -gt 0 ]]; then
    sleep "$POLL_SECONDS"
  fi
done

if [[ "$stop_submissions" -eq 1 ]]; then
  for ((i = 0; i < TASK_COUNT; i++)); do
    if [[ "${TASK_STATES[$i]}" == "pending" ]]; then
      TASK_STATES[$i]="skipped"
    fi
  done
fi

{
  printf 'task_id\tgroup\tkind\tconfig\tsplit\tseed\tattempts\tstate\n'
  for ((i = 0; i < TASK_COUNT; i++)); do
    if [[ "${TASK_STATES[$i]}" == "failed" || "${TASK_STATES[$i]}" == "skipped" ]]; then
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$i" "${TASK_GROUPS[$i]}" "${TASK_KINDS[$i]}" "${TASK_CONFIGS[$i]}" \
        "${TASK_SPLITS[$i]}" "${TASK_SEEDS[$i]}" "${TASK_ATTEMPTS[$i]}" "${TASK_STATES[$i]}"
    fi
  done
} > "$FAILED_REPORT"

GROUPS=()
for ((i = 0; i < TASK_COUNT; i++)); do
  found=0
  for group in "${GROUPS[@]}"; do
    if [[ "$group" == "${TASK_GROUPS[$i]}" ]]; then
      found=1
      break
    fi
  done
  if [[ "$found" -eq 0 ]]; then
    GROUPS+=("${TASK_GROUPS[$i]}")
  fi
done

for group in "${GROUPS[@]}"; do
  run_dirs=()
  for ((i = 0; i < TASK_COUNT; i++)); do
    if [[ "${TASK_GROUPS[$i]}" == "$group" && "${TASK_STATES[$i]}" == "done" ]]; then
      run_dirs+=("${TASK_RUN_DIRS[$i]}")
    fi
  done

  if [[ "${#run_dirs[@]}" -eq 0 ]]; then
    log "INFO" "Skipping summary for ${group}; no successful runs"
    continue
  fi

  summary_output="${SUMMARY_DIR}/slurm_${MANIFEST_NAME}_${group}_summary.json"
  log "INFO" "Writing summary for ${group} to ${summary_output}"
  "$PYTHON_EXE" -m src.summarize --runs "${run_dirs[@]}" --output "$summary_output"
done

if [[ "$permanent_failures" -gt 0 ]]; then
  log_error "Completed with ${permanent_failures} permanent failure(s). See ${FAILED_REPORT}"
  exit 1
fi

log "INFO" "All Slurm masking tasks completed successfully"
