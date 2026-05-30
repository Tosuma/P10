#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 <Hugin J. Zachariasen, Magnus H. Jensen, Tobias S. Madsen>.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
COMBINED_JOB_SCRIPT="${PROJECT_ROOT}/scripts/mami/hpfat/stage1/eval/predict_eval_job.sh"

cd "${PROJECT_ROOT}" || exit 1

sri_path="${PROJECT_ROOT}/data/sri-lanka-aligned/"
weedy_path="${PROJECT_ROOT}/data/data/WeedyRice/"

RETRY_TEXT="Could not lookup the current user"
MAX_RETRIES=50
MAX_PARALLEL=8

mkdir -p "${PROJECT_ROOT}/logs/hpfat/eval"
mkdir -p "${PROJECT_ROOT}/logs/hpfat/inference"
mkdir -p "${PROJECT_ROOT}/logs/hpfat/predict_eval"

if ! command -v sbatch >/dev/null 2>&1; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') :: Error: 'sbatch' command not found."
    exit 1
fi

if ! command -v squeue >/dev/null 2>&1; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') :: Error: 'squeue' command not found."
    exit 1
fi

log_job_failure() {
    job_id="$1"
    out_file="$2"
    err_file="$3"

    job_state=""
    if command -v sacct >/dev/null 2>&1; then
        job_state="$(sacct -j "${job_id}" --format=State --noheader 2>/dev/null | head -n 1 | xargs)"
    fi

    if [ -n "${job_state}" ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') :: Slurm state for job ${job_id}: ${job_state}"
    fi

    if [ -f "${err_file}" ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') :: Last lines from ${err_file}:"
        tail -n 40 "${err_file}"
    fi

    if [ -f "${out_file}" ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') :: Last lines from ${out_file}:"
        tail -n 40 "${out_file}"
    fi
}

process_pair() {
    ndre="$1"
    ndvi="$2"

    dir_name="${PROJECT_ROOT}/results/hpfat/andhra-stage1---Weedy-Rice/re_${ndre}_vi_${ndvi}"
    model_name="${PROJECT_ROOT}/checkpoints/hpfat/stage1/re_${ndre}_vi_${ndvi}/hpfat-andhra-re_${ndre}-vi_${ndvi}_stage1_best.pth"

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

    attempt=0

    while true; do
        job_id=$(
            sbatch "${COMBINED_JOB_SCRIPT}" \
                --model_name "${model_name}" \
                --dir_name "${dir_name}" \
                --truth "${weedy_path}" \
                --type "Weedy-Rice" \
                | grep -o '[0-9]\+'
        )

        if [ -z "${job_id}" ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') :: Failed to submit predict+eval job for ndre=${ndre}, ndvi=${ndvi}"

            if [ "$attempt" -ge "$MAX_RETRIES" ]; then
                echo "$(date '+%Y-%m-%d %H:%M:%S') :: Reached max retries for predict+eval submit. Stopping pair."
                return 1
            fi

            attempt=$((attempt + 1))
            sleep 60
            continue
        fi

        echo "$(date '+%Y-%m-%d %H:%M:%S') :: Starting predict+eval job ${job_id} ndre=${ndre}, ndvi=${ndvi}"

        while squeue --me | grep -q "$job_id"; do
            sleep 10
        done

        out_file="${PROJECT_ROOT}/logs/hpfat/predict_eval/predict_eval_mami_${job_id}.out"
        err_file="${PROJECT_ROOT}/logs/hpfat/predict_eval/predict_eval_mami_${job_id}.err"

        if [ ! -f "$err_file" ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') :: Err file $err_file did not appear. Retrying predict+eval..."

            if [ "$attempt" -ge "$MAX_RETRIES" ]; then
                echo "$(date '+%Y-%m-%d %H:%M:%S') :: Reached max retries for predict+eval ndre=${ndre}, ndvi=${ndvi}."
                return 1
            fi

            attempt=$((attempt + 1))
            sleep 60
            continue
        fi

        first_line="$(head -n 1 "$err_file")"

        if grep -qF "$RETRY_TEXT" "$err_file"; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') :: Predict+eval job ${job_id} hit retryable error: '${first_line}'"

            if [ "$attempt" -ge "$MAX_RETRIES" ]; then
                echo "$(date '+%Y-%m-%d %H:%M:%S') :: Reached max retries for predict+eval ndre=${ndre}, ndvi=${ndvi}."
                return 1
            fi

            attempt=$((attempt + 1))
            sleep 10
            continue
        fi

        if [ ! -e "$results" ] || [ ! -e "$summary" ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') :: Predict+eval finished, but result files are missing for ndre=${ndre}, ndvi=${ndvi}"
            echo "$(date '+%Y-%m-%d %H:%M:%S') :: Expected:"
            echo "  $results"
            echo "  $summary"
            log_job_failure "${job_id}" "${out_file}" "${err_file}"

            # Do not blindly retry non-retryable failures; surface the root cause.
            return 1
        fi

        echo "$(date '+%Y-%m-%d %H:%M:%S') :: Finished predict+eval successfully for ndre=${ndre}, ndvi=${ndvi}"
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

available_models=0

for ndre in $(seq -f "%.1f" 0.0 0.1 1.0); do
    for ndvi in $(seq -f "%.1f" 0.0 0.1 1.0); do
        model_name="${PROJECT_ROOT}/checkpoints/hpfat/stage1/re_${ndre}_vi_${ndvi}/hpfat-andhra-re_${ndre}-vi_${ndvi}_stage1_best.pth"
        if [ -f "${model_name}" ]; then
            available_models=$((available_models + 1))
        fi
    done
done

if [ "${available_models}" -eq 0 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') :: Error: No model checkpoints found under ${PROJECT_ROOT}/checkpoints/hpfat/stage1/"
    echo "$(date '+%Y-%m-%d %H:%M:%S') :: Nothing to evaluate; expected files like:"
    echo "  ${PROJECT_ROOT}/checkpoints/hpfat/stage1/re_0.0_vi_0.0/hpfat-andhra-re_0.0-vi_0.0_stage1_best.pth"
    exit 1
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') :: Found ${available_models} checkpoint(s) to evaluate."

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
