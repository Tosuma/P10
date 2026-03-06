#!/bin/bash
mkdir -p logs/vi

LR="4e-4"
MRAE="1.0"
MODEL_BASE_NAME="vi-kaz"
BASE_DIR="vi"

# 0.0 -> 1.0 in 0.1 increments, formatted to one decimal
for ndre in $(seq -f "%.1f" 0.4 0.1 0.6); do
    for ndvi in $(seq -f "%.1f" 0.4 0.1 0.6); do
        echo "$(date '+%Y-%m-%d %H:%M:%d') :: Starting iteration ndvi: ${ndvi}, ndre: ${ndre}"
        
        # Make dir_name unique per run to avoid overwriting outputs
        DIR_NAME="${BASE_DIR}/ndvi_${ndre}_ndre_${ndvi}"
        MODEL_NAME="${MODEL_BASE_NAME}-vi_${ndre}-re_${ndvi}"

        job_id=$(sbatch scripts/mami/train_mami_job.sh \
            --lr "${LR}" \
            --loss_mrae_w "${MRAE}" \
            --loss_ndre_w "${ndre}" \
            --loss_ndvi_w "${ndvi}" \
            --dir_name "${DIR_NAME}" \
            --model_name "${MODEL_NAME}" \
            | grep -o '[0-9]\+')
        
        while squeue --me | grep -q "$job_id"; do
            # echo "Job $job_id still running... sleeping 5 minutes"
            sleep 300
        done
    done
done
