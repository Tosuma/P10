#!/bin/bash
mkdir -p logs/vi

LR="1e-5"
MRAE="1.0"
MODEL_BASE_NAME="vi-weedy"
BASE_DIR="vi/stage2"

RETRY_TEXT="Could not lookup the current user"
MAX_RETRIES=50

chosen_models=(
    "checkpoints/vi/finals/vi-kaz-re_0.4-vi_0.0_stage1_best.pth"
)

for train_model in "${chosen_models[@]}"; do
    for ndre in $(seq -f "%.1f" 0.5 0.1 1.0); do
        for ndvi in $(seq -f "%.1f" 0.0 0.1 1.0); do
            DIR_NAME="${BASE_DIR}/re_${ndre}_vi_${ndvi}"
            MODEL_NAME="${MODEL_BASE_NAME}-re_${ndre}-vi_${ndvi}"

            attempt=0

            if [[ ("$ndre" == "0.5" && ("$ndvi" == "0.0" || "$ndvi" == "0.1" || "$ndvi" == "0.2" || "$ndvi" == "0.3" || "$ndvi" == "0.4" || "$ndvi" == "0.5")) ]]; then
                echo "$(date '+%Y-%m-%d %H:%M:%S') :: skipped ndre=${ndre}, ndvi=${ndvi}"
                continue
            fi

            while true; do
                job_id=$(
                    sbatch scripts/mami/vi/stage2/train_mami_job.sh \
                        --train_model "${train_model}" \
                        --lr "${LR}" \
                        --loss_mrae_w "${MRAE}" \
                        --loss_ndre_w "${ndre}" \
                        --loss_ndvi_w "${ndvi}" \
                        --dir_name "${DIR_NAME}" \
                        --model_name "${MODEL_NAME}" \
                    | grep -o '[0-9]\+'
                )

                echo "$(date '+%Y-%m-%d %H:%M:%S') :: Starting job ${job_id} ndre: ${ndre}, ndvi: ${ndvi}"

                while squeue --me | grep -q "$job_id"; do
                    sleep 10
                done

                err_file="logs/vi/train_mami_${job_id}.err"

                sleep 5
                if [ ! -f "${err_file}" ]; then
                    echo "$(date '+%Y-%m-%d %H:%M:%S') :: Err file ${err_file} did not appear. Retrying..."

                    if [ "${attempt}" -ge "${MAX_RETRIES}" ]; then
                        echo "$(date '+%Y-%m-%d %H:%M:%S') :: Reached max retries (${MAX_RETRIES}) for ndre=${ndre}, ndvi=${ndvi}. Stopping."
                        exit 1
                    fi

                    attempt=$((attempt + 1))
                    sleep 60
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

                rm -f "logs/vi/train_mami_${job_id}.out"
                break
            done
        done
    done
done