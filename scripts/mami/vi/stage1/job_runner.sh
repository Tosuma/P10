#!/bin/bash
mkdir -p logs/vi
LR="1e-5"
MRAE="1.0"
MODEL_BASE_NAME="stage1-andhra"
BASE_DIR="andhra/reproduce"

RETRY_TEXT="Could not lookup the current user"
MAX_RETRIES=20


ndre=0.0
ndvi=0.0

for number in {1..9}; do
    epoch=$((number*30))
    DIR_NAME="${BASE_DIR}/re_${ndre}_vi_${ndvi}_stage1"
    MODEL_NAME="${MODEL_BASE_NAME}-re_${ndre}-vi_${ndvi}"
    attempt=5
    echo "${epoch}"
    while true; do
        job_id=$(
            sbatch scripts/mami/train_mami_andhra_job.sh \
                --lr "${LR}" \
                --loss_mrae_w "${MRAE}" \
                --loss_ndre_w "${ndre}" \
                --loss_ndvi_w "${ndvi}" \
                --dir_name "${DIR_NAME}" \
                --model_name "${MODEL_NAME}" \
		--epoch "${epoch}" \
            | grep -o '[0-9]\+'
        )

        echo "$(date '+%Y-%m-%d %H:%M:%d') :: Starting job ${job_id} ndre: ${ndre}, ndvi: ${ndvi}"

            err_file="logs/vi/train_mami_${job_id}.err"
            sleep 5
        while squeue --me | grep -q "$job_id"; do
            # echo "Job $job_id still running... sleeping 5 minutes"
            sleep 60
        done # sleep

        err_file="logs/vi/train_mami_${job_id}.err"
        sleep 60

        if [ ! -f "${err_file}" ]; then
	    echo "$(date '+%Y-%m-%d %H:%M:%S') :: Err file ${err_file} did not appear. Retrying..."
	    if [ "${attempt}" -ge "${MAX_RETRIES}" ]; then 
	        echo "$(date '+%Y-%m-%d %H:%M:%S') :: Reached max retries (${MAX_RETRIES})"
		    exit 1
	    fi
	    attemp=$((attempt + 1))
            continue
	fi
            
        first_line=""
        if [ -f "${err_file}" ]; then
            first_line="$(head -n 1 "${err_file}")"
        fi

        if [[ "${first_line}" == *"${RETRY_TEXT}"* ]]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') :: Job ${job_id} hit retryable error: '${first_line}'"
	fi

        echo "$(date '+%Y-%m-%d %H:%M:%S') :: Finished job ${job_id} successfully for ndre=${ndre}, ndvi=${ndvi}"
    done
done
