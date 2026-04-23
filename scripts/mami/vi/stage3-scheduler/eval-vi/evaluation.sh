#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 <Hugin J. Zachariasen, Magnus H. Jensen, Tobias S. Madsen>.


sri_path="./data/sri-lanka-aligned/"
weedy_path="./data/WeedyRice/"

RETRY_TEXT="Could not lookup the current user"
MAX_RETRIES=50

mkdir -p logs/eval
mkdir -p logs/inference

for ndre in $(seq -f "%.1f" 0.0 0.1 0.5); do
  for ndvi in $(seq -f "%.1f" 0.0 0.1 1.0); do
    dir_name="results/vi/stage3-scheduler---sri-lanka/re_${ndre}_vi_${ndvi}"
    model_name="./checkpoints/vi/finals/stage3-scheduler/vi-weedy-scheduler-re_${ndre}-vi_${ndvi}_stage3_best.pth"

    attempt=0

    results="${dir_name}/results.json"
    summary="${dir_name}/summary.json"
    if [ -e "$results" ] && [ -e "$summary" ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%d') :: RE: $ndre, VI: $ndvi has already been evaluated"
        continue
    fi

    # predict
    while true; do
        job_id=$(
            sbatch scripts/mami/vi/stage3/eval-vi/predict_job.sh \
            --model_name "${model_name}" \
            --dir_name "${dir_name}" \
            --truth "${sri_path}" \
            --type "Sri-Lanka" \
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

        echo "$(date '+%Y-%m-%d %H:%M:%d') :: Starting inference job ${job_id} ndre: ${ndre}, ndvi: ${ndvi}"

        while squeue --me | grep -q "$job_id"; do
            # echo "Job $job_id still running... sleeping 5 minutes"
            sleep 10
        done # sleep

        err_file="logs/inference/pred_mami_${job_id}.err"
        
        if [ ! -f "$err_file" ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') :: Err file $err_file did not appear. Retrying..."

            if [ "$attempt" -ge "$MAX_RETRIES" ]; then
                echo "$(date '+%Y-%m-%d %H:%M:%S') :: Reached max retries ($MAX_RETRIES) for ndre=$ndre, ndvi=$ndvi. Stopping."
                exit 1
            fi

            attempt=$((attempt + 1))
            continue
        fi

        first_line="$(head -n 1 "$err_file")"

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

        echo "$(date '+%Y-%m-%d %H:%M:%S') :: Finished inference job ${job_id} successfully for ndre=${ndre}, ndvi=${ndvi}"

        # remove out dir
        rm "logs/inference/pred_mami_${job_id}.out"
        break # the retry loop
    done # retry loop

    # evaluate
    while true; do
        job_id=$(
            sbatch scripts/mami/vi/stage3/eval-vi/eval_job.sh \
            --model_name "${model_name}" \
            --dir_name "${dir_name}" \
            --truth "${sri_path}" \
            --type "Sri-Lanka" \
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

        echo "$(date '+%Y-%m-%d %H:%M:%d') :: Starting evaluation job ${job_id} ndre: ${ndre}, ndvi: ${ndvi}"

        while squeue --me | grep -q "$job_id"; do
            # echo "Job $job_id still running... sleeping 5 minutes"
            sleep 10
        done # sleep

        err_file="logs/eval/eval_mami_${job_id}.err"

        if [ ! -f "$err_file" ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') :: Err file $err_file did not appear. Retrying..."

            if [ "$attempt" -ge "$MAX_RETRIES" ]; then
                echo "$(date '+%Y-%m-%d %H:%M:%S') :: Reached max retries ($MAX_RETRIES) for ndre=$ndre, ndvi=$ndvi. Stopping."
                exit 1
            fi

            attempt=$((attempt + 1))
            continue
        fi

        first_line="$(head -n 1 "$err_file")"

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

        echo "$(date '+%Y-%m-%d %H:%M:%S') :: Finished evaluation job ${job_id} successfully for ndre=${ndre}, ndvi=${ndvi}"

        # move evaluation logs
        rm "logs/eval/eval_mami_${job_id}.out"

        # delete data folder
        rm -r "${dir_name}/data"
        
        break # the retry loop
    done # retry loop
  done # ndvi loop
done # ndre loop
