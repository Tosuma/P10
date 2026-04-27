#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 <Hugin J. Zachariasen, Magnus H. Jensen, Tobias S. Madsen>.

DEFAULT_MODEL_NAME="./checkpoints/vi/finals/stage3/vi-weedy-re_0.0-vi_0.0_stage3_best.pth"
DEFAULT_DIR_NAME="results/vi/stage3---sri-lanka/inference"
DEFAULT_TRUTH="./data/sri-lanka-aligned/"
DEFAULT_TYPE="Sri-Lanka"

RETRY_TEXT="Could not lookup the current user"
MAX_RETRIES=50

model_name="$DEFAULT_MODEL_NAME"
dir_name="$DEFAULT_DIR_NAME"
truth="$DEFAULT_TRUTH"
type="$DEFAULT_TYPE"

usage() {
    cat <<EOF
Usage: $0 [options]

Runs the inference portion of the stage3 eval flow with the same retry guards.

Options:
  --model_name, --model <path>  Model checkpoint to use
  --dir_name, --output <path>   Output directory root; predictions are written to <dir>/data
  --truth <path>                Dataset root path
  --type <name>                 Dataset type (Sri-Lanka, Kazakhstan, Weedy-Rice)
  --help                        Show this help text
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_name|--model)
            [[ $# -ge 2 ]] || { echo "Error: $1 requires a value" >&2; exit 2; }
            model_name="$2"
            shift 2
            ;;
        --dir_name|--output)
            [[ $# -ge 2 ]] || { echo "Error: $1 requires a value" >&2; exit 2; }
            dir_name="$2"
            shift 2
            ;;
        --truth)
            [[ $# -ge 2 ]] || { echo "Error: $1 requires a value" >&2; exit 2; }
            truth="$2"
            shift 2
            ;;
        --type)
            [[ $# -ge 2 ]] || { echo "Error: $1 requires a value" >&2; exit 2; }
            type="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            echo "Unexpected positional arg: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

mkdir -p logs/inference

attempt=0

while true; do
    job_id=$(
        sbatch scripts/mami/inference/predict_job.sh \
        --model_name "${model_name}" \
        --dir_name "${dir_name}" \
        --truth "${truth}" \
        --type "${type}" \
        | grep -o '[0-9]\+'
    )

    if [[ -z "$job_id" ]]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') :: Failed to parse sbatch job id" >&2
        exit 1
    fi

    echo "$(date '+%Y-%m-%d %H:%M:%S') :: Starting inference job ${job_id} model=${model_name} output=${dir_name}"

    while squeue --me | grep -q "$job_id"; do
        sleep 10
    done

    err_file="logs/inference/pred_mami_${job_id}.err"

    if [[ ! -f "$err_file" ]]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') :: Err file $err_file did not appear. Retrying..."

        if [[ "$attempt" -ge "$MAX_RETRIES" ]]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') :: Reached max retries ($MAX_RETRIES). Stopping."
            exit 1
        fi

        attempt=$((attempt + 1))
        continue
    fi

    first_line="$(head -n 1 "$err_file")"

    if grep -qF "$RETRY_TEXT" "$err_file"; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') :: Job ${job_id} hit retryable error: '${first_line}'"

        if [[ "${attempt}" -ge "${MAX_RETRIES}" ]]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') :: Reached max retries (${MAX_RETRIES}). Stopping."
            exit 1
        fi

        attempt=$((attempt + 1))
        echo "$(date '+%Y-%m-%d %H:%M:%S') :: Retrying model=${model_name} output=${dir_name}"
        sleep 60
        continue
    fi

    echo "$(date '+%Y-%m-%d %H:%M:%S') :: Finished inference job ${job_id} successfully model=${model_name} output=${dir_name}"

    out_file="logs/inference/pred_mami_${job_id}.out"
    if [[ -f "$out_file" ]]; then
        rm "$out_file"
    fi

    break
done
