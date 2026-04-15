#!/bin/bash
mkdir -p logs/vi

NDRE="0.0"
NDVI="0.0"
MRAE="1.0"

RETRY_TEXT="Could not lookup the current user"
MAX_RETRIES=50

run_stage() {
    local stage="$1"
    local train_script="$2"
    local lr="$3"
    local dir_name="$4"
    local model_name="$5"
    local train_model="${6:-}"

    local final_model="checkpoints/${dir_name}/all-models/${model_name}_${stage}_final.pth"
    local final_model_line="Final model: ${final_model}"
    local attempt=0

    if grep -RqsF -- "${final_model_line}" logs/vi; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') :: ${stage} ndre=${NDRE}, ndvi=${NDVI} has already finished"
        return 0
    fi

    while true; do
        if [ -n "${train_model}" ]; then
            job_id=$(
                sbatch "${train_script}" \
                    --train_model "${train_model}" \
                    --lr "${lr}" \
                    --loss_mrae_w "${MRAE}" \
                    --loss_ndre_w "${NDRE}" \
                    --loss_ndvi_w "${NDVI}" \
                    --dir_name "${dir_name}" \
                    --model_name "${model_name}" \
                | grep -o '[0-9]\+'
            )
        else
            job_id=$(
                sbatch "${train_script}" \
                    --lr "${lr}" \
                    --loss_mrae_w "${MRAE}" \
                    --loss_ndre_w "${NDRE}" \
                    --loss_ndvi_w "${NDVI}" \
                    --dir_name "${dir_name}" \
                    --model_name "${model_name}" \
                | grep -o '[0-9]\+'
            )
        fi

        if [ -z "${job_id}" ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') :: Empty job id returned for ${stage} ndre=${NDRE}, ndvi=${NDVI}. Retrying..."

            if [ "${attempt}" -ge "${MAX_RETRIES}" ]; then
                echo "$(date '+%Y-%m-%d %H:%M:%S') :: Reached max retries (${MAX_RETRIES}) for ${stage} ndre=${NDRE}, ndvi=${NDVI}. Stopping."
                exit 1
            fi

            attempt=$((attempt + 1))
            sleep 60
            continue
        fi

        echo "$(date '+%Y-%m-%d %H:%M:%S') :: Starting ${stage} job ${job_id} ndre=${NDRE}, ndvi=${NDVI}"

        while squeue --me | grep -q "${job_id}"; do
            sleep 10
        done

        err_file="logs/vi/train_mami_${job_id}.err"

        if [ ! -f "${err_file}" ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') :: Err file ${err_file} did not appear for ${stage}. Retrying..."

            if [ "${attempt}" -ge "${MAX_RETRIES}" ]; then
                echo "$(date '+%Y-%m-%d %H:%M:%S') :: Reached max retries (${MAX_RETRIES}) for ${stage} ndre=${NDRE}, ndvi=${NDVI}. Stopping."
                exit 1
            fi

            attempt=$((attempt + 1))
            sleep 60
            continue
        fi

        first_line="$(head -n 1 "${err_file}")"

        if grep -qF "${RETRY_TEXT}" "${err_file}"; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') :: ${stage} job ${job_id} hit retryable error: '${first_line}'"

            if [ "${attempt}" -ge "${MAX_RETRIES}" ]; then
                echo "$(date '+%Y-%m-%d %H:%M:%S') :: Reached max retries (${MAX_RETRIES}) for ${stage} ndre=${NDRE}, ndvi=${NDVI}. Stopping."
                exit 1
            fi

            attempt=$((attempt + 1))
            echo "$(date '+%Y-%m-%d %H:%M:%S') :: Retrying ${stage} ndre=${NDRE}, ndvi=${NDVI}"
            sleep 60
            continue
        fi

        echo "$(date '+%Y-%m-%d %H:%M:%S') :: Finished ${stage} job ${job_id} successfully for ndre=${NDRE}, ndvi=${NDVI}"

        rm -f "logs/vi/train_mami_${job_id}.out"
        break
    done
}

resolve_checkpoint() {
    local stage="$1"
    local dir_name="$2"
    local model_name="$3"

    local best_model="checkpoints/${dir_name}/${model_name}_${stage}_best.pth"
    local final_model="checkpoints/${dir_name}/all-models/${model_name}_${stage}_final.pth"

    if [ -f "${best_model}" ]; then
        echo "${best_model}"
        return 0
    fi

    if [ -f "${final_model}" ]; then
        echo "${final_model}"
        return 0
    fi

    echo "$(date '+%Y-%m-%d %H:%M:%S') :: Could not find ${stage} checkpoint. Checked '${best_model}' and '${final_model}'." >&2
    exit 1
}

STAGE1_DIR="vi/reproduce/stage1/re_${NDRE}_vi_${NDVI}"
STAGE1_MODEL="vi-kaz-re_${NDRE}-vi_${NDVI}"
run_stage \
    "stage1" \
    "scripts/mami/vi/reproduce/train_mami_job.sh" \
    "4e-4" \
    "${STAGE1_DIR}" \
    "${STAGE1_MODEL}"
STAGE1_CHECKPOINT="$(resolve_checkpoint "stage1" "${STAGE1_DIR}" "${STAGE1_MODEL}")"

STAGE2_DIR="vi/reproduce/stage2/re_${NDRE}_vi_${NDVI}"
STAGE2_MODEL="vi-weedy-re_${NDRE}-vi_${NDVI}"
run_stage \
    "stage2" \
    "scripts/mami/vi/stage2/train_mami_job.sh" \
    "1e-5" \
    "${STAGE2_DIR}" \
    "${STAGE2_MODEL}" \
    "${STAGE1_CHECKPOINT}"
STAGE2_CHECKPOINT="$(resolve_checkpoint "stage2" "${STAGE2_DIR}" "${STAGE2_MODEL}")"

STAGE3_DIR="vi/reproduce/stage3/re_${NDRE}_vi_${NDVI}"
STAGE3_MODEL="vi-weedy-re_${NDRE}-vi_${NDVI}"
run_stage \
    "stage3" \
    "scripts/mami/vi/stage3/train_mami_job.sh" \
    "1e-7" \
    "${STAGE3_DIR}" \
    "${STAGE3_MODEL}" \
    "${STAGE2_CHECKPOINT}"

echo "$(date '+%Y-%m-%d %H:%M:%S') :: Finished chained reproduce run for ndre=${NDRE}, ndvi=${NDVI}"
