#!/bin/bash

set -euo pipefail

MANIFEST=""
CHUNK_SIZE=12
MAX_RETRIES=50
POLL_SECONDS=10
DRY_RUN=0

LR="1e-5"
LOSS_MRAE_W="1.0"
STAGE2_EPOCHS="300"
STAGE2_DATA_PATH="data/data/sri-lanka-aligned"
STAGE2_DATA_TYPE="Sri-Lanka"
SINGULARITY_IMAGE="/ceph/container/pytorch/pytorch_26.02.sif"
VENV_ACTIVATE="p10_venv/bin/activate"
MAX_SUBMIT_RETRIES=5
SUBMIT_RETRY_SLEEP=60

SBATCH_SCRIPT="scripts/mami/hpfat/stage2/train_mami_batch_job.sh"
LOG_ROOT="logs/hpfat/stage2"
STATUS_DIR=""
TASK_LOG_DIR="${LOG_ROOT}/tasks"
CONTROLLER_LOG="${LOG_ROOT}/controller.log"

trim() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
}

status_value() {
    local file="$1"
    local key="$2"
    awk -F= -v k="$key" '$1 == k { sub("^[^=]*=", ""); print; exit }' "$file"
}

log() {
    local level="$1"
    shift
    local msg="$*"
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    printf '%s :: [%s] %s\n' "$ts" "$level" "$msg" | tee -a "$CONTROLLER_LOG"
}

usage() {
    cat <<'EOF'
Usage:
  bash scripts/mami/hpfat/stage2/job_runner.sh --manifest scripts/mami/hpfat/stage2/tasks_manifest.csv

Options:
  --manifest PATH
  --chunk-size N                  Number of tasks per sbatch (default: 4)
  --max-retries N                 Retries after initial failure per task (default: 50)
  --poll-seconds N                Seconds between squeue polls (default: 10)
  --dry-run                       Validate and print chunking without submitting jobs
  --status-dir PATH               Override persistent status directory
  --task-log-dir PATH             Override per-task log directory
  --lr VALUE                      Stage2 learning rate (default: 1e-5)
  --loss_mrae_w VALUE             Stage2 MRAE loss weight (default: 1.0)
  --stage2_epochs N               Stage2 epochs (default: 300)
  --stage2_data_path PATH         Stage2 data path (default: data/WeedyRice)
  --stage2_data_type VALUE        Stage2 data type (default: Weedy-Rice)
  --singularity-image PATH        Container image path
  --venv-activate PATH            Venv activation path inside container
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest)          MANIFEST="$2"; shift 2 ;;
        --chunk-size)        CHUNK_SIZE="$2"; shift 2 ;;
        --max-retries)       MAX_RETRIES="$2"; shift 2 ;;
        --poll-seconds)      POLL_SECONDS="$2"; shift 2 ;;
        --dry-run)           DRY_RUN=1; shift ;;
        --status-dir)        STATUS_DIR="$2"; shift 2 ;;
        --task-log-dir)      TASK_LOG_DIR="$2"; shift 2 ;;
        --lr)                LR="$2"; shift 2 ;;
        --loss_mrae_w)       LOSS_MRAE_W="$2"; shift 2 ;;
        --stage2_epochs)     STAGE2_EPOCHS="$2"; shift 2 ;;
        --stage2_data_path)  STAGE2_DATA_PATH="$2"; shift 2 ;;
        --stage2_data_type)  STAGE2_DATA_TYPE="$2"; shift 2 ;;
        --singularity-image) SINGULARITY_IMAGE="$2"; shift 2 ;;
        --venv-activate)     VENV_ACTIVATE="$2"; shift 2 ;;
        --help|-h)           usage; exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

: "${MANIFEST:?Missing --manifest}"

if [[ ! -f "$MANIFEST" ]]; then
    echo "Manifest not found: $MANIFEST" >&2
    exit 1
fi

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
    echo "Batch script not found: $SBATCH_SCRIPT" >&2
    exit 1
fi

if [[ "$CHUNK_SIZE" -lt 1 || "$CHUNK_SIZE" -gt 4 ]]; then
    echo "--chunk-size must be between 1 and 4." >&2
    exit 2
fi

if [[ "$MAX_RETRIES" -lt 0 ]]; then
    echo "--max-retries must be >= 0." >&2
    exit 2
fi

if [[ "$POLL_SECONDS" -lt 1 ]]; then
    echo "--poll-seconds must be >= 1." >&2
    exit 2
fi

manifest_stem="$(basename "$MANIFEST")"
manifest_stem="${manifest_stem%.*}"
if [[ -z "$STATUS_DIR" ]]; then
    STATUS_DIR="${LOG_ROOT}/status/${manifest_stem}"
fi

mkdir -p "$LOG_ROOT" "$STATUS_DIR" "$TASK_LOG_DIR"
touch "$CONTROLLER_LOG"

declare -a TASK_IDS=()
declare -A MANIFEST_STATUS=()
declare -A SEEN_IDS=()

MANIFEST_HEADER=""
DELIM=""
declare -A COL_IDX=()
line_no=0

while IFS= read -r line || [[ -n "$line" ]]; do
    line_no=$((line_no + 1))
    line="${line%$'\r'}"
    stripped="$(trim "$line")"
    if [[ -z "$stripped" || "$stripped" == \#* ]]; then
        continue
    fi

    if [[ -z "$MANIFEST_HEADER" ]]; then
        MANIFEST_HEADER="$line"
        if [[ "$line" == *","* ]]; then
            DELIM=","
        else
            DELIM=$'\t'
        fi

        IFS="$DELIM" read -r -a header_cols <<< "$line"
        for i in "${!header_cols[@]}"; do
            col="$(trim "${header_cols[$i]}")"
            COL_IDX["$col"]="$i"
        done

        required=(task_id train_model loss_ndre_w loss_ndvi_w dir_name model_name)
        for col in "${required[@]}"; do
            if [[ -z "${COL_IDX[$col]+x}" ]]; then
                echo "Manifest missing required column '${col}'." >&2
                exit 2
            fi
        done
        continue
    fi

    IFS="$DELIM" read -r -a cols <<< "$line"

    task_id="$(trim "${cols[${COL_IDX[task_id]}]:-}")"
    train_model="$(trim "${cols[${COL_IDX[train_model]}]:-}")"
    ndre="$(trim "${cols[${COL_IDX[loss_ndre_w]}]:-}")"
    ndvi="$(trim "${cols[${COL_IDX[loss_ndvi_w]}]:-}")"
    dir_name="$(trim "${cols[${COL_IDX[dir_name]}]:-}")"
    model_name="$(trim "${cols[${COL_IDX[model_name]}]:-}")"

    if [[ -z "$task_id" || -z "$train_model" || -z "$ndre" || -z "$ndvi" || -z "$dir_name" || -z "$model_name" ]]; then
        echo "Malformed manifest line ${line_no}: required value missing." >&2
        exit 2
    fi

    if [[ -n "${SEEN_IDS[$task_id]+x}" ]]; then
        echo "Duplicate task_id '${task_id}' in manifest (line ${line_no})." >&2
        exit 2
    fi
    SEEN_IDS["$task_id"]=1

    TASK_IDS+=("$task_id")
    if [[ -n "${COL_IDX[status]+x}" ]]; then
        MANIFEST_STATUS["$task_id"]="$(trim "${cols[${COL_IDX[status]}]:-}")"
    else
        MANIFEST_STATUS["$task_id"]=""
    fi
done < "$MANIFEST"

if [[ -z "$MANIFEST_HEADER" || "${#TASK_IDS[@]}" -eq 0 ]]; then
    echo "Manifest must contain header and at least one task row." >&2
    exit 2
fi

max_attempts=$((MAX_RETRIES + 1))

declare -A TASK_DONE=()
declare -A TASK_ATTEMPTS=()

for task_id in "${TASK_IDS[@]}"; do
    TASK_DONE["$task_id"]=0
    TASK_ATTEMPTS["$task_id"]=0

    manifest_status="${MANIFEST_STATUS[$task_id]:-}"
    if [[ "$manifest_status" == "success" ]]; then
        TASK_DONE["$task_id"]=1
    fi

    status_file="${STATUS_DIR}/task_${task_id}.status"
    if [[ -f "$status_file" ]]; then
        previous_state="$(status_value "$status_file" "state")"
        previous_attempt="$(status_value "$status_file" "attempt")"
        if [[ "$previous_attempt" =~ ^[0-9]+$ ]]; then
            TASK_ATTEMPTS["$task_id"]="$previous_attempt"
        fi
        if [[ "$previous_state" == "success" ]]; then
            TASK_DONE["$task_id"]=1
        fi
    fi
done

log "INFO" "Loaded ${#TASK_IDS[@]} tasks from ${MANIFEST}"
log "INFO" "chunk_size=${CHUNK_SIZE}, max_retries=${MAX_RETRIES}, poll_seconds=${POLL_SECONDS}, dry_run=${DRY_RUN}"
log "INFO" "status_dir=${STATUS_DIR}"

if [[ "$DRY_RUN" -eq 1 ]]; then
    pending_for_dry=()
    for task_id in "${TASK_IDS[@]}"; do
        if [[ "${TASK_DONE[$task_id]}" -eq 0 ]]; then
            pending_for_dry+=("$task_id")
        fi
    done

    log "INFO" "Dry run: ${#pending_for_dry[@]} pending tasks."
    batch_count=0
    for ((i=0; i<${#pending_for_dry[@]}; i+=CHUNK_SIZE)); do
        chunk=( "${pending_for_dry[@]:i:CHUNK_SIZE}" )
        batch_count=$((batch_count + 1))
        log "INFO" "Dry run batch ${batch_count}: ${chunk[*]}"
    done
    log "INFO" "Dry run complete."
    exit 0
fi

if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch was not found on PATH." >&2
    exit 1
fi
if ! command -v squeue >/dev/null 2>&1; then
    echo "squeue was not found on PATH." >&2
    exit 1
fi

while true; do
    pending=()
    exhausted=()
    for task_id in "${TASK_IDS[@]}"; do
        if [[ "${TASK_DONE[$task_id]}" -eq 1 ]]; then
            continue
        fi
        attempts="${TASK_ATTEMPTS[$task_id]}"
        if [[ "$attempts" -ge "$max_attempts" ]]; then
            exhausted+=("$task_id")
        else
            pending+=("$task_id")
        fi
    done

    if [[ "${#pending[@]}" -eq 0 ]]; then
        if [[ "${#exhausted[@]}" -gt 0 ]]; then
            log "ERROR" "Tasks exhausted retries: ${exhausted[*]}"
            exit 1
        fi
        log "INFO" "All tasks completed successfully."
        exit 0
    fi

    chunk=( "${pending[@]:0:CHUNK_SIZE}" )
    declare -A EXPECTED_ATTEMPTS=()
    attempt_map_items=()
    for task_id in "${chunk[@]}"; do
        next_attempt=$((TASK_ATTEMPTS[$task_id] + 1))
        TASK_ATTEMPTS["$task_id"]="$next_attempt"
        EXPECTED_ATTEMPTS["$task_id"]="$next_attempt"
        attempt_map_items+=( "${task_id}:${next_attempt}" )
    done

    task_ids_csv="$(IFS=,; echo "${chunk[*]}")"
    attempt_map_csv="$(IFS=,; echo "${attempt_map_items[*]}")"

    submit_try=0
    job_id=""
    while true; do
        submit_try=$((submit_try + 1))
        submit_output="$(
            sbatch "$SBATCH_SCRIPT" \
                --manifest "$MANIFEST" \
                --task-ids "$task_ids_csv" \
                --attempt-map "$attempt_map_csv" \
                --status-dir "$STATUS_DIR" \
                --task-log-dir "$TASK_LOG_DIR" \
                --lr "$LR" \
                --loss_mrae_w "$LOSS_MRAE_W" \
                --stage2_epochs "$STAGE2_EPOCHS" \
                --stage2_data_path "$STAGE2_DATA_PATH" \
                --stage2_data_type "$STAGE2_DATA_TYPE" \
                --singularity-image "$SINGULARITY_IMAGE" \
                --venv-activate "$VENV_ACTIVATE" \
            2>&1
        )"
        job_id="$(echo "$submit_output" | grep -o '[0-9]\+' | tail -n 1 || true)"
        if [[ -n "$job_id" ]]; then
            break
        fi

        if [[ "$submit_try" -ge "$MAX_SUBMIT_RETRIES" ]]; then
            log "ERROR" "Could not submit batch after ${MAX_SUBMIT_RETRIES} tries: ${submit_output}"
            exit 1
        fi
        log "ERROR" "sbatch returned no job id (try ${submit_try}/${MAX_SUBMIT_RETRIES}): ${submit_output}"
        sleep "$SUBMIT_RETRY_SLEEP"
    done

    log "INFO" "Submitted job ${job_id} for tasks: ${chunk[*]}"

    while squeue -h -j "$job_id" | grep -q .; do
        sleep "$POLL_SECONDS"
    done

    log "INFO" "Job ${job_id} finished. Validating task statuses."

    for task_id in "${chunk[@]}"; do
        expected_attempt="${EXPECTED_ATTEMPTS[$task_id]}"
        status_file="${STATUS_DIR}/task_${task_id}.status"

        if [[ ! -f "$status_file" ]]; then
            log "ERROR" "Task ${task_id}: missing status file after job ${job_id}."
            continue
        fi

        state="$(status_value "$status_file" "state")"
        attempt="$(status_value "$status_file" "attempt")"
        retryable="$(status_value "$status_file" "retryable")"
        exit_code="$(status_value "$status_file" "exit_code")"
        message="$(status_value "$status_file" "message")"

        if [[ "$attempt" != "$expected_attempt" ]]; then
            log "ERROR" "Task ${task_id}: stale status file attempt=${attempt}, expected=${expected_attempt}."
            continue
        fi

        if [[ "$state" == "success" ]]; then
            TASK_DONE["$task_id"]=1
            log "INFO" "Task ${task_id}: success on attempt ${attempt}."
        else
            log "ERROR" "Task ${task_id}: failed on attempt ${attempt} (exit=${exit_code}, retryable=${retryable}) ${message}"
        fi
    done
done
