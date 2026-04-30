#!/bin/bash
mkdir -p logs/hpfat

LR="4e-4"
MRAE="1.0"
MODEL_BASE_NAME="hpfat-andhra"
BASE_DIR="hpfat/stage1"

RETRY_TEXT="Could not lookup the current user"
MAX_RETRIES=50

NDVI_VALUES=("0.0" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0")
FULL_NDRE_VALUES=("0.3" "0.4")
FIRST_BOUNDARY_NDRE="0.2"
FIRST_BOUNDARY_NDVI_VALUES=("0.8" "0.9" "1.0")
SECOND_BOUNDARY_NDRE="0.5"
SECOND_BOUNDARY_NDVI_VALUES=("0.0" "0.1" "0.2" "0.3" "0.4")

run_job() {
    local ndre="$1"
    local ndvi="$2"
    local dir_name="${BASE_DIR}/re_${ndre}_vi_${ndvi}"
    local model_name="${MODEL_BASE_NAME}-re_${ndre}-vi_${ndvi}"
    local final_model_line="Final model: checkpoints/${dir_name}"
    local attempt=0

    if grep -RqsF -- "${final_model_line}" logs/hpfat; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') :: RE: $ndre, VI: $ndvi has already been evaluated"
        return
    fi

    while true; do
        job_id=$(
            sbatch scripts/mami/hpfat/stage1/train_mami_job.sh \
                --lr "${LR}" \
                --loss_mrae_w "${MRAE}" \
                --loss_ndre_w "${ndre}" \
                --loss_ndvi_w "${ndvi}" \
                --dir_name "${dir_name}" \
                --model_name "${model_name}" \
            | grep -o '[0-9]\+'
        )

        if [ -z "${job_id}" ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') :: Empty job id returned for ndre=${ndre}, ndvi=${ndvi}. Retrying..."

            if [ "${attempt}" -ge "${MAX_RETRIES}" ]; then
                echo "$(date '+%Y-%m-%d %H:%M:%S') :: Reached max retries (${MAX_RETRIES}) for ndre=${ndre}, ndvi=${ndvi}. Stopping."
                exit 1
            fi

            attempt=$((attempt + 1))
            sleep 60
            continue
        fi

        echo "$(date '+%Y-%m-%d %H:%M:%S') :: Starting job ${job_id} ndre: ${ndre}, ndvi: ${ndvi}"

        while squeue --me | grep -q "$job_id"; do
            sleep 10
        done

        err_file="logs/hpfat/train_mami_${job_id}.err"

        if [ ! -f "${err_file}" ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') :: Err file ${err_file} did not appear. Retrying..."

            if [ "${attempt}" -ge "${MAX_RETRIES}" ]; then
                echo "$(date '+%Y-%m-%d %H:%M:%S') :: Reached max retries (${MAX_RETRIES}) for ndre=${ndre}, ndvi=${ndvi}. Stopping."
                exit 1
            fi

            attempt=$((attempt + 1))
            continue
        fi

        first_line="$(head -n 1 "${err_file}")"

        if grep -qF "$RETRY_TEXT" "$err_file"; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') :: Job ${job_id} hit retryable error: '${first_line}'"

            if [ "${attempt}" -ge "${MAX_RETRIES}" ]; then
                echo "$(date '+%Y-%m-%d %H:%M:%S') :: Reached max retries (${MAX_RETRIES}) for ndre=${ndre}, ndvi=${ndvi}. Stopping."
                exit 1
            fi

            attempt=$((attempt + 1))
            echo "$(date '+%Y-%m-%d %H:%M:%S') :: Retrying ndre=${ndre}, ndvi=${ndvi}"
            sleep 60
            continue
        fi

        echo "$(date '+%Y-%m-%d %H:%M:%S') :: Finished job ${job_id} successfully for ndre=${ndre}, ndvi=${ndvi}"

        rm -f "logs/hpfat/train_mami_${job_id}.out"
        break
    done
}

for ndvi in "${FIRST_BOUNDARY_NDVI_VALUES[@]}"; do
    run_job "${FIRST_BOUNDARY_NDRE}" "${ndvi}"
done

for ndre in "${FULL_NDRE_VALUES[@]}"; do
    for ndvi in "${NDVI_VALUES[@]}"; do
        run_job "${ndre}" "${ndvi}"
    done
done

for ndvi in "${SECOND_BOUNDARY_NDVI_VALUES[@]}"; do
    run_job "${SECOND_BOUNDARY_NDRE}" "${ndvi}"
done
