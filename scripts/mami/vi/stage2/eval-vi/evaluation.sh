#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 <Hugin J. Zachariasen, Magnus H. Jensen, Tobias S. Madsen>.


sri_path="./data/sri-lanka-aligned/"
weedy_path="./data/WeedyRice/"

RETRY_TEXT="Could not lookup the current user"
MAX_RETRIES=50
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREDICT_SCRIPT="${SCRIPT_DIR}/predict_job.sh"
EVAL_SCRIPT="${SCRIPT_DIR}/eval_job.sh"

mkdir -p logs/eval
mkdir -p logs/inference

for ndre in $(seq -f "%.1f" 0.0 0.1 0.5); do
  for ndvi in $(seq -f "%.1f" 0.0 0.1 1.0); do
    dir_name="results/vi/stage2---sri-lanka/re_${ndre}_vi_${ndvi}"
    model_name="./checkpoints/vi/finals/stage2/vi-weedy-re_${ndre}-vi_${ndvi}_stage2_best.pth"

    attempt=0

    results="${dir_name}/results.json"
    summary="${dir_name}/summary.json"
    if [ -e "$results" ] && [ -e "$summary" ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%d') :: RE: $ndre, VI: $ndvi has already been evaluated"
        continue
    fi

    # predict
    while true; do
        attempt="${attempt:-0}"
        job_id="$(date '+%Y%m%d_%H%M%S')_$$_pred_re_${ndre}_vi_${ndvi}_attempt_${attempt}"
        out_file="logs/inference/pred_mami_${job_id}.out"
        err_file="logs/inference/pred_mami_${job_id}.err"

        echo "$(date '+%Y-%m-%d %H:%M:%S') :: Starting Inference srun ${job_id} ndre: ${ndre}, ndvi: ${ndvi}"

        srun \
            --job-name=eval_mami \
            --output="${out_file}" \
            --error="${err_file}" \
            --nodes=1 \
            --ntasks=1 \
            --mem=24G \
            --cpus-per-task=15 \
            --gres=gpu:1 \
            --time=12:00:00 \
            bash "${PREDICT_SCRIPT}" \
            --model_name "${model_name}" \
            --dir_name "${dir_name}" \
            --truth "${sri_path}" \
            --type "Sri-Lanka" \
            2>> "${err_file}"
        exit_code=$?
        
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
            echo "$(date '+%Y-%m-%d %H:%M:%S') :: srun ${job_id} hit retryable error: '${first_line}'"

            if [ "${attempt}" -ge "${MAX_RETRIES}" ]; then
                echo "$(date '+%Y-%m-%d %H:%M:%S') :: Reached max retries (${MAX_RETRIES}) for ndre=${ndre}, ndvi=${ndvi}. Stopping."
                exit 1
            fi

            attempt=$((attempt + 1))
            echo "$(date '+%Y-%m-%d %H:%M:%S') :: Retrying ndre=${ndre}, ndvi=${ndvi}"
            sleep 60
            continue
        fi

        if [ "${exit_code}" -ne 0 ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') :: Inference srun ${job_id} failed with exit code ${exit_code}. See ${err_file}."
            exit "${exit_code}"
        fi

        echo "$(date '+%Y-%m-%d %H:%M:%S') :: Finished inference srun ${job_id} successfully for ndre=${ndre}, ndvi=${ndvi}"

        # remove out dir
        rm "logs/inference/pred_mami_${job_id}.out"
        break # the retry loop
    done # retry loop

    # evaluate
    while true; do
        attempt="${attempt:-0}"
        job_id="$(date '+%Y%m%d_%H%M%S')_$$_eval_re_${ndre}_vi_${ndvi}_attempt_${attempt}"
        out_file="logs/eval/eval_mami_${job_id}.out"
        err_file="logs/eval/eval_mami_${job_id}.err"

        echo "$(date '+%Y-%m-%d %H:%M:%S') :: Starting Evaluation srun ${job_id} ndre: ${ndre}, ndvi: ${ndvi}"

        srun \
            --job-name=eval_mami \
            --output="${out_file}" \
            --error="${err_file}" \
            --nodes=1 \
            --ntasks=1 \
            --mem=24G \
            --cpus-per-task=15 \
            --gres=gpu:1 \
            --time=12:00:00 \
            bash "${EVAL_SCRIPT}" \
            --model_name "${model_name}" \
            --dir_name "${dir_name}" \
            --truth "${sri_path}" \
            --type "Sri-Lanka" \
            2>> "${err_file}"
        exit_code=$?

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
            echo "$(date '+%Y-%m-%d %H:%M:%S') :: srun ${job_id} hit retryable error: '${first_line}'"

            if [ "${attempt}" -ge "${MAX_RETRIES}" ]; then
                echo "$(date '+%Y-%m-%d %H:%M:%S') :: Reached max retries (${MAX_RETRIES}) for ndre=${ndre}, ndvi=${ndvi}. Stopping."
                exit 1
            fi

            attempt=$((attempt + 1))
            echo "$(date '+%Y-%m-%d %H:%M:%S') :: Retrying ndre=${ndre}, ndvi=${ndvi}"
            sleep 10
            continue
        fi

        if [ "${exit_code}" -ne 0 ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') :: Evaluation srun ${job_id} failed with exit code ${exit_code}. See ${err_file}."
            exit "${exit_code}"
        fi

        echo "$(date '+%Y-%m-%d %H:%M:%S') :: Finished evaluation srun ${job_id} successfully for ndre=${ndre}, ndvi=${ndvi}"

        # move evaluation logs
        rm "logs/eval/eval_mami_${job_id}.out"

        # delete data folder
        rm -r "${dir_name}/data"
        
        break # the retry loop
    done # retry loop
  done # ndvi loop
done # ndre loop
