#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 <Hugin J. Zachariasen, Magnus H. Jensen, Tobias S. Madsen>.

set -u

sri_path="./data/sri-lanka-aligned/"
weedy_path="./data/data/WeedyRice/"

RETRY_TEXT="Could not lookup the current user"
MAX_RETRIES=50
MAX_PARALLEL=8

mkdir -p logs/hpfat/eval
mkdir -p logs/hpfat/inference

process_pair() {
    ndre="$1"
    ndvi="$2"

    dir_name="results/hpfat/andhra-stage1---Weedy-Rice/re_${ndre}_vi_${ndvi}"
    model_name="./checkpoints/hpfat/stage1/re_${ndre}_vi_${ndvi}/hpfat-andhra-re_${ndre}-vi_${ndvi}_stage1_best.pth"

    results="${dir_name}/results.json"
    summary="${dir_name}/summary.json"

    if [ ! -f "$model_name" ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') :: Model file does not exist: $model_name, skipping"
        return 0
    fi

    if [ -e "$results" ] && [ -e "$summary" ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') :: RE: $ndre, VI: $ndvi has already been evaluated"
        return 0
    fi

    mkdir -p "$dir_name"

    ###########################################################################
    # Predict
    ###########################################################################

    attempt=0

    while true; do
        job_id=$(
            sbatch scripts/mami/hpfat/stage1/eval/predict_job.sh \
                --model_name "${model_name}" \
                --dir_name "${dir_name}" \
                --truth "${weedy_path}" \
                --type "Weedy-Rice" \
                | grep -o '[0-9]\+'
        )

        if [ -z "${job_id}" ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') :: Failed to submit inference job for ndre=${ndre}, ndvi=${ndvi}"

            if [ "$attempt" -ge "$MAX_RETRIES" ]; then
                echo "$(date '+%Y-%m-%d %H:%M:%S') :: Reached max retries for inference submit. Stopping pair."
                return 1
            fi

            attempt=$((attempt + 1))
            sleep 60
            continue
        fi

        echo "$(date '+%Y-%m-%d %H:%M:%S') :: Starting inference job ${job_id} ndre=${ndre}, ndvi=${ndvi}"

        while squeue --me | grep -q "$job_id"; do
            sleep 10
        done

        err_file="logs/hpfat/inference/pred_mami_${job_id}.err"

        if [ ! -f "$err_file" ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') :: Err file $err_file did not appear. Retrying inference..."

            if [ "$attempt" -ge "$MAX_RETRIES" ]; then
                echo "$(date '+%Y-%m-%d %H:%M:%S') :: Reached max retries for inference ndre=${ndre}, ndvi=${ndvi}."
                return 1
            fi

            attempt=$((attempt + 1))
            sleep 60
            continue
        fi

        first_line="$(head -n 1 "$err_file")"

        if grep -qF "$RETRY_TEXT" "$err_file"; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') :: Inference job ${job_id} hit retryable error: '${first_line}'"

            if [ "$attempt" -ge "$MAX_RETRIES" ]; then
                echo "$(date '+%Y-%m-%d %H:%M:%S') :: Reached max retries for inference ndre=${ndre}, ndvi=${ndvi}."
                return 1
            fi

            attempt=$((attempt + 1))
            sleep 60
            continue
        fi

        echo "$(date '+%Y-%m-%d %H:%M:%S') :: Finished inference job ${job_id} successfully for ndre=${ndre}, ndvi=${ndvi}"
        break
    done

    ###########################################################################
    # Evaluate
    ###########################################################################

    attempt=0

    while true; do
        job_id=$(
            sbatch scripts/mami/hpfat/stage1/eval/eval_job.sh \
                --model_name "${model_name}" \
                --dir_name "${dir_name}" \
                --truth "${weedy_path}" \
                --type "Weedy-Rice" \
                | grep -o '[0-9]\+'
        )

        if [ -z "${job_id}" ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') :: Failed to submit evaluation job for ndre=${ndre}, ndvi=${ndvi}"

            if [ "$attempt" -ge "$MAX_RETRIES" ]; then
                echo "$(date '+%Y-%m-%d %H:%M:%S') :: Reached max retries for evaluation submit. Stopping pair."
                return 1
            fi

            attempt=$((attempt + 1))
            sleep 60
            continue
        fi

        echo "$(date '+%Y-%m-%d %H:%M:%S') :: Starting evaluation job ${job_id} ndre=${ndre}, ndvi=${ndvi}"

        while squeue --me | grep -q "$job_id"; do
            sleep 10
        done

        err_file="logs/hpfat/eval/eval_mami_${job_id}.err"

        if [ ! -f "$err_file" ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') :: Err file $err_file did not appear. Retrying evaluation..."

            if [ "$attempt" -ge "$MAX_RETRIES" ]; then
                echo "$(date '+%Y-%m-%d %H:%M:%S') :: Reached max retries for evaluation ndre=${ndre}, ndvi=${ndvi}."
                return 1
            fi

            attempt=$((attempt + 1))
            sleep 60
            continue
        fi

        first_line="$(head -n 1 "$err_file")"

        if grep -qF "$RETRY_TEXT" "$err_file"; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') :: Evaluation job ${job_id} hit retryable error: '${first_line}'"

            if [ "$attempt" -ge "$MAX_RETRIES" ]; then
                echo "$(date '+%Y-%m-%d %H:%M:%S') :: Reached max retries for evaluation ndre=${ndre}, ndvi=${ndvi}."
                return 1
            fi

            attempt=$((attempt + 1))
            sleep 10
            continue
        fi

        if [ ! -e "$results" ] || [ ! -e "$summary" ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') :: Evaluation finished, but result files are missing for ndre=${ndre}, ndvi=${ndvi}"
            echo "$(date '+%Y-%m-%d %H:%M:%S') :: Expected:"
            echo "  $results"
            echo "  $summary"

            if [ "$attempt" -ge "$MAX_RETRIES" ]; then
                echo "$(date '+%Y-%m-%d %H:%M:%S') :: Reached max retries waiting for result files."
                return 1
            fi

            attempt=$((attempt + 1))
            sleep 30
            continue
        fi

        echo "$(date '+%Y-%m-%d %H:%M:%S') :: Finished evaluation successfully for ndre=${ndre}, ndvi=${ndvi}"
        echo "$(date '+%Y-%m-%d %H:%M:%S') :: Produced result files:"
        echo "  $results"
        echo "  $summary"

        rm -rf "${dir_name}/data"

        break
    done

    return 0
}

###############################################################################
# Run up to 8 ndre/ndvi pairs at once
###############################################################################

active_jobs=0
failed=0

for ndre in $(seq -f "%.1f" 0.0 0.1 1.0); do
    for ndvi in $(seq -f "%.1f" 0.0 0.1 1.0); do

        process_pair "$ndre" "$ndvi" &

        active_jobs=$((active_jobs + 1))

        if [ "$active_jobs" -ge "$MAX_PARALLEL" ]; then
            if ! wait -n; then
                failed=1
            fi
            active_jobs=$((active_jobs - 1))
        fi

    done
done

while [ "$active_jobs" -gt 0 ]; do
    if ! wait -n; then
        failed=1
    fi
    active_jobs=$((active_jobs - 1))
done

if [ "$failed" -ne 0 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') :: One or more ndre/ndvi jobs failed."
    exit 1
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') :: All jobs completed successfully."