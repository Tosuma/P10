#!/bin/bash
mkdir -p logs/vi
LR="4e-4"
MRAE="1.0"
MODEL_BASE_NAME="vi-kaz"
BASE_DIR="vi"
RETRY_TEXT="Could not lookup the current user"
MAX_RETRIES=10
for ndre in $(seq -f "%.1f" 0.7 0.1 1.0); do
    for ndvi in $(seq -f "%.1f" 0.0 0.1 1.0); do
        DIR_NAME="${BASE_DIR}/re_${ndre}_vi_${ndvi}"
        MODEL_NAME="${MODEL_BASE_NAME}-re_${ndre}-vi_${ndvi}"
        attempt=1
        while true; do
            job_id=$(
                sbatch scripts/mami/train_mami_job.sh \
                    --lr "${LR}" \
                    --loss_mrae_w "${MRAE}" \
                    --loss_ndre_w "${ndre}" \
                    --loss_ndvi_w "${ndvi}" \
                    --dir_name "${DIR_NAME}" \
                    --model_name "${MODEL_NAME}" \
                | grep -o '[0-9]\+'
            )
            echo "$(date '+%Y-%m-%d %H:%M:%d') :: Starting job ${job_id} ndre: ${ndre}, ndvi: ${ndvi}"
            while squeue --me | grep -q "$job_id"; do
                # echo "Job $job_id still running... sleeping 5 minutes"
                sleep 10
            done # sleep
            err_file="logs/vi/train_mami/${job_id}.err"
            sleep 5

            first_line=""
            if [ -f "${err_file}" ]; then
                first_line="$(head -n 1 "${err_file}")"
            fi
            if [[ "${first_line}" == *"${RETRY_TEXT}"* ]]; then
                echo "$(date '+%Y-%m-%d %H:%M:%S') :: Job ${job_id} hit retryable error: '${first_line}'"
                if [ "${attempt}" -ge "${MAX_RETRIES}" ]; then
                    echo "$(date '+%Y-%m-%d %H:%M:%S') :: Reached max retries (${MAX_RETRIES}) for ndre=${ndre}, ndvi=${ndvi}. Stopping."
                    exit 1
                fi
                attempt=$((attempt + 1))
                echo "$(date '+%Y-%m-%d %H:%M:%S') :: Retrying ndre=${ndre}, ndvi=${ndvi}"
                sleep 10
                continue
            fi
            echo "$(date '+%Y-%m-%d %H:%M:%S') :: Finished job ${job_id} successfully for ndre=${ndre}, ndvi=${ndvi}"
            break # the retry loop
        done # retry loop
    done # ndvi loop
done # ndre loop