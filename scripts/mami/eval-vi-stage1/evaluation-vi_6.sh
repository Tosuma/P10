#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 <Hugin J. Zachariasen, Magnus H. Jensen, Tobias S. Madsen>.


sri_path="./data/sri-lanka-aligned/"
weedy_path="./data/WeedyRice/"

RETRY_TEXT="Could not lookup the current user"
MAX_RETRIES=10

mkdir -p logs/eval
mkdir -p logs/inference

for ndre in $(seq -f "%.1f" 0.8 0.1 0.8); do
  for ndvi in $(seq -f "%.1f" 0.0 0.1 1.0); do
    dir_name="results/vi/re_${ndre}_vi_${ndvi}_stage1"
    model_name="./checkpoints/vi/finals/vi-kaz-re_${ndre}-vi_${ndvi}_stage1_best.pth"

    # predict
    while true; do
        job_id=$(
            sbatch scripts/mami/eval-vi-stage1/predict_job.sh \
            --model_name "${model_name}" \
            --dir_name "${dir_name}---weedy-rice" \
            --truth "${weedy_path}" \
            --type "Weedy-Rice" \
            | grep -o '[0-9]\+'
        )

        echo "$(date '+%Y-%m-%d %H:%M:%d') :: Starting job ${job_id} ndre: ${ndre}, ndvi: ${ndvi}"

        while squeue --me | grep -q "$job_id"; do
            # echo "Job $job_id still running... sleeping 5 minutes"
            sleep 10
        done # sleep

        err_file="logs/inference/pred_mami_${job_id}.err"
        sleep 5

        first_line=""
        if [ -f "${err_file}" ]; then
            first_line="$(head -n 1 "${err_file}")"
        fi

        if grep -qF "$RETRY_TEXT" "$err_file"; then
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

        # move prediction logs
        mv "logs/inference/pred_mami_${job_id}.err" "logs/eval/pred_re_${ndre}_vi_${ndvi}_stage1.err"
        echo "$(date '+%Y-%m-%d %H:%M:%S') :: Renamed 'vi/pred_mami_${job_id}.err' to 'eval/re_${ndre}_vi_${ndvi}.err'"
        
        mv "logs/inference/pred_mami_${job_id}.out" "logs/eval/pred_re_${ndre}_vi_${ndvi}_stage1.out"
        echo "$(date '+%Y-%m-%d %H:%M:%S') :: Renamed 'vi/pred_mami_${job_id}.out' to 'eval/re_${ndre}_vi_${ndvi}.out'"
        break # the retry loop
    done # retry loop

    # evaluate
    while true; do
        job_id=$(
            sbatch scripts/mami/eval-vi-stage1/eval_job.sh \
            --model_name "${model_name}" \
            --dir_name "${dir_name}---weedy-rice" \
            --truth "${weedy_path}" \
            --type "Weedy-Rice" \
            | grep -o '[0-9]\+'
        )

        echo "$(date '+%Y-%m-%d %H:%M:%d') :: Starting job ${job_id} ndre: ${ndre}, ndvi: ${ndvi}"

        while squeue --me | grep -q "$job_id"; do
            # echo "Job $job_id still running... sleeping 5 minutes"
            sleep 10
        done # sleep

        err_file="logs/inference/eval_mami_${job_id}.err"
        sleep 5

        first_line=""
        if [ -f "${err_file}" ]; then
            first_line="$(head -n 1 "${err_file}")"
        fi

        if grep -qF "$RETRY_TEXT" "$err_file"; then
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

        # move evaluation logs
        mv "logs/inference/eval_mami_${job_id}.err" "logs/eval/eval_re_${ndre}_vi_${ndvi}_stage1.err"
        echo "$(date '+%Y-%m-%d %H:%M:%S') :: Renamed 'vi/eval_mami_${job_id}.err' to 'eval/re_${ndre}_vi_${ndvi}.err'"
        
        mv "logs/inference/eval_mami_${job_id}.out" "logs/eval/eval_re_${ndre}_vi_${ndvi}_stage1.out"
        echo "$(date '+%Y-%m-%d %H:%M:%S') :: Renamed 'vi/eval_mami_${job_id}.out' to 'eval/re_${ndre}_vi_${ndvi}.out'"
        break # the retry loop
    done # retry loop
  done # ndvi loop
done # ndre loop