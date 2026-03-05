#!/bin/bash
mkdir -p logs

LR="4e-4"
MRAE="1.0"
MODEL_BASE_NAME="vi-stage1-kaz"
BASE_DIR="vi"

# 0.0 -> 1.0 in 0.1 increments, formatted to one decimal
for ndre in $(seq -f "%.1f" 0.0 0.1 1.0); do
    for ndvi in $(seq -f "%.1f" 0.0 0.1 1.0); do
        date
        echo "Starting iteration ndvi: ${ndvi}, ndre: ${ndre}"
        
        job_id=$(sbatch test.sh \
        --lr "hello" \
        | grep -o '[0-9]\+')

        echo "Submitted job with ID: ${job_id}"

        # Make dir_name unique per run to avoid overwriting outputs
        DIR_NAME="${BASE_DIR}/ndvi_${ndre}_ndre_${ndvi}"
        MODEL_NAME="${MODEL_BASE_NAME}-vi_${ndre}-re_${ndvi}"

        # job_id=$(sbatch train_mami.sh \
        #     --lr "${LR}" \
        #     --loss_mrae_w "${MRAE}" \
        #     --loss_ndre_w "${ndre}" \
        #     --loss_ndvi_w "${ndvi}" \
        #     --dir_name "${DIR_NAME}" \
        #     --model_name "${MODEL_NAME}" \
        #     | grep -o '[0-9]\+')
        
        while squeue --me | grep -q "$job_id"; do
            echo "Job $job_id still running... sleeping 10 seconds"
            sleep 10
        done
    done
done