#!/bin/bash
#SBATCH --job-name=train_mami_batch
#SBATCH --output=logs/hpfat/stage3/train_mami_batch_%j.out
#SBATCH --error=logs/hpfat/stage3/train_mami_batch_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00

set -euo pipefail

MANIFEST=""
TASK_IDS_CSV=""
ATTEMPT_MAP=""
STATUS_DIR="logs/hpfat/stage3/status/default"
TASK_LOG_DIR="logs/hpfat/stage3/tasks"
LR="1e-7"
LOSS_MRAE_W="1.0"
STAGE3_EPOCHS="300"
STAGE3_DATA_PATH="data/WeedyRice"
STAGE3_DATA_TYPE="Weedy-Rice"
SINGULARITY_IMAGE="/ceph/container/pytorch/pytorch_26.02.sif"
VENV_ACTIVATE="p10_venv/bin/activate"
RETRY_TEXT="Could not lookup the current user"

usage() {
    cat <<'EOF'
Usage:
  sbatch scripts/mami/hpfat/stage3/train_mami_batch_job.sh \
    --manifest scripts/mami/hpfat/stage3/tasks_manifest.csv \
    --task-ids task_001,task_002,task_003,task_004 \
    --attempt-map task_001:1,task_002:1,task_003:1,task_004:1

Options:
  --manifest PATH
  --task-ids CSV                     Comma-separated task IDs (max 4)
  --attempt-map CSV                  Comma-separated pairs task_id:attempt
  --status-dir PATH                  Directory for per-task status artifacts
  --task-log-dir PATH                Directory for per-task logs
  --lr VALUE                         Stage3 learning rate (default: 1e-5)
  --loss_mrae_w VALUE                Stage3 MRAE loss weight (default: 1.0)
  --stage3_epochs N                  Stage3 epochs (default: 300)
  --stage3_data_path PATH            Stage3 data path (default: data/WeedyRice)
  --stage3_data_type VALUE           Stage3 data type (default: Weedy-Rice)
  --singularity-image PATH           Container image path
  --venv-activate PATH               Venv activation path inside container
EOF
}

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

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest)          MANIFEST="$2"; shift 2 ;;
        --task-ids)          TASK_IDS_CSV="$2"; shift 2 ;;
        --attempt-map)       ATTEMPT_MAP="$2"; shift 2 ;;
        --status-dir)        STATUS_DIR="$2"; shift 2 ;;
        --task-log-dir)      TASK_LOG_DIR="$2"; shift 2 ;;
        --lr)                LR="$2"; shift 2 ;;
        --loss_mrae_w)       LOSS_MRAE_W="$2"; shift 2 ;;
        --stage3_epochs)     STAGE3_EPOCHS="$2"; shift 2 ;;
        --stage3_data_path)  STAGE3_DATA_PATH="$2"; shift 2 ;;
        --stage3_data_type)  STAGE3_DATA_TYPE="$2"; shift 2 ;;
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
: "${TASK_IDS_CSV:?Missing --task-ids}"

if [[ ! -f "$MANIFEST" ]]; then
    echo "Manifest not found: $MANIFEST" >&2
    exit 1
fi

mkdir -p "$(dirname "$STATUS_DIR")" "$STATUS_DIR" "$TASK_LOG_DIR" "logs/hpfat/stage3"

declare -a TASK_IDS=()
IFS=',' read -r -a TASK_IDS <<< "$TASK_IDS_CSV"

if [[ "${#TASK_IDS[@]}" -eq 0 ]]; then
    echo "No task IDs were provided in --task-ids." >&2
    exit 2
fi

declare -A ATTEMPT_BY_TASK=()
if [[ -n "$ATTEMPT_MAP" ]]; then
    IFS=',' read -r -a attempt_items <<< "$ATTEMPT_MAP"
    for item in "${attempt_items[@]}"; do
        task_id="${item%%:*}"
        attempt="${item#*:}"
        task_id="$(trim "$task_id")"
        attempt="$(trim "$attempt")"
        if [[ -n "$task_id" && "$attempt" =~ ^[0-9]+$ ]]; then
            ATTEMPT_BY_TASK["$task_id"]="$attempt"
        fi
    done
fi

MANIFEST_HEADER=""
DELIM=""
declare -A COL_IDX=()
declare -A TRAIN_MODEL_BY_ID=()
declare -A NDRE_BY_ID=()
declare -A NDVI_BY_ID=()
declare -A DIR_BY_ID=()
declare -A MODEL_BY_ID=()
declare -A SEEN_IDS=()

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

        IFS="$DELIM" read -r -a header_cols <<< "$MANIFEST_HEADER"
        for i in "${!header_cols[@]}"; do
            col="$(trim "${header_cols[$i]}")"
            COL_IDX["$col"]="$i"
        done

        required_cols=(task_id train_model loss_ndre_w loss_ndvi_w dir_name model_name)
        for col in "${required_cols[@]}"; do
            if [[ -z "${COL_IDX[$col]+x}" ]]; then
                echo "Manifest missing required column '${col}' in header." >&2
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

    TRAIN_MODEL_BY_ID["$task_id"]="$train_model"
    NDRE_BY_ID["$task_id"]="$ndre"
    NDVI_BY_ID["$task_id"]="$ndvi"
    DIR_BY_ID["$task_id"]="$dir_name"
    MODEL_BY_ID["$task_id"]="$model_name"
done < "$MANIFEST"

if [[ -z "$MANIFEST_HEADER" ]]; then
    echo "Manifest is empty: $MANIFEST" >&2
    exit 2
fi

hostname
date
echo "Running task IDs: ${TASK_IDS[*]}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
GPUS="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-1}}"

batch_failed=0

write_status() {
    local task_id="$1"
    local state="$2"
    local retryable="$3"
    local attempt="$4"
    local exit_code="$5"
    local message="$6"
    local task_log="$7"

    local status_file="${STATUS_DIR}/task_${task_id}.status"
    local tmp_file="${status_file}.tmp"

    {
        echo "task_id=${task_id}"
        echo "state=${state}"
        echo "retryable=${retryable}"
        echo "attempt=${attempt}"
        echo "exit_code=${exit_code}"
        echo "message=${message}"
        echo "job_id=${SLURM_JOB_ID:-}"
        echo "task_log=${task_log}"
        echo "updated_at=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    } > "$tmp_file"
    mv "$tmp_file" "$status_file"
}

for raw_task_id in "${TASK_IDS[@]}"; do
    task_id="$(trim "$raw_task_id")"
    if [[ -z "$task_id" ]]; then
        continue
    fi

    task_attempt="${ATTEMPT_BY_TASK[$task_id]:-1}"
    task_log="${TASK_LOG_DIR}/task_${task_id}_attempt_${task_attempt}.log"
    : > "$task_log"

    if [[ -z "${TRAIN_MODEL_BY_ID[$task_id]+x}" ]]; then
        echo "Task '${task_id}' was not found in manifest." | tee -a "$task_log"
        write_status "$task_id" "failed" "false" "$task_attempt" "2" "task_id not found in manifest" "$task_log"
        batch_failed=1
        continue
    fi

    train_model="${TRAIN_MODEL_BY_ID[$task_id]}"
    ndre="${NDRE_BY_ID[$task_id]}"
    ndvi="${NDVI_BY_ID[$task_id]}"
    dir_name="${DIR_BY_ID[$task_id]}"
    model_name="${MODEL_BY_ID[$task_id]}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Task ${task_id} attempt ${task_attempt}: starting" | tee -a "$task_log"
    echo "model=${train_model} ndre=${ndre} ndvi=${ndvi} dir=${dir_name} name=${model_name}" | tee -a "$task_log"

    set +e
    singularity exec --nv \
        -B /ceph/project/tbd/data/:/ceph/project/tbd/data \
        "$SINGULARITY_IMAGE" \
        /bin/bash -lc "source '$VENV_ACTIVATE' && \
            python -u -m torch.distributed.run \
                --standalone \
                --nproc_per_node=${GPUS} \
                mami/mami.py \
                    --stage3_model '${train_model}' \
                    --stage3_data_path '${STAGE3_DATA_PATH}' \
                    --stage3_data_type '${STAGE3_DATA_TYPE}' \
                    --stage3_epochs '${STAGE3_EPOCHS}' \
                    --stage3_lr '${LR}' \
                    --stage3_loss_mrae_w '${LOSS_MRAE_W}' \
                    --stage3_loss_ndvi_w '${ndvi}' \
                    --stage3_loss_ndre_w '${ndre}' \
                    --dir_name '${dir_name}' \
                    --model_name '${model_name}' \
                    --cluster" \
        2>&1 | tee -a "$task_log"
    cmd_exit="${PIPESTATUS[0]}"
    set -e

    if [[ "$cmd_exit" -eq 0 ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Task ${task_id}: success" | tee -a "$task_log"
        write_status "$task_id" "success" "false" "$task_attempt" "$cmd_exit" "completed" "$task_log"
        continue
    fi

    retryable="false"
    if grep -qF "$RETRY_TEXT" "$task_log"; then
        retryable="true"
    fi

    message="task failed"
    if [[ "$retryable" == "true" ]]; then
        message="task failed with retryable cluster error"
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Task ${task_id}: failed (exit=${cmd_exit}, retryable=${retryable})" | tee -a "$task_log"
    write_status "$task_id" "failed" "$retryable" "$task_attempt" "$cmd_exit" "$message" "$task_log"
    batch_failed=1
done

date
exit "$batch_failed"
