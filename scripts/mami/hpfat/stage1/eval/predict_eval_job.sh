#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 <Hugin J. Zachariasen, Magnus H. Jensen, Tobias S. Madsen>.

#SBATCH --job-name=pred_eval_mami
#SBATCH --output=logs/hpfat/predict_eval/predict_eval_mami_%j.out
#SBATCH --error=logs/hpfat/predict_eval/predict_eval_mami_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT=""

# Under Slurm, BASH_SOURCE[0] can be a staged temporary copy. Prefer submit dir.
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
elif [ -d "${PWD}" ] && [ -f "${PWD}/mami/inference.py" ]; then
    PROJECT_ROOT="${PWD}"
else
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
fi

model_name=""
dir_name=""
truth=""
type=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dir_name)     dir_name="$2"; shift 2 ;;
        --model_name)   model_name="$2"; shift 2 ;;
        --truth)        truth="$2"; shift 2 ;;
        --type)         type="$2"; shift 2 ;;
        -*)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
        *)
            echo "Unexpected positional arg: $1" >&2
            exit 2
            ;;
    esac
done

: "${dir_name:?Missing --dir_name}"
: "${model_name:?Missing --model_name}"
: "${type:?Missing --type}"
: "${truth:?Missing --truth}"

if [ ! -f "${PROJECT_ROOT}/mami/inference.py" ] || [ ! -f "${PROJECT_ROOT}/mami/evaluation.py" ]; then
    echo "Could not resolve project root with mami scripts." >&2
    echo "Resolved PROJECT_ROOT: ${PROJECT_ROOT}" >&2
    echo "SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR:-<unset>}" >&2
    exit 1
fi

mkdir -p "${dir_name}/data"

echo "=== Beginning predictions and evaluation ==="
echo "model: ${model_name}"
echo "truth: ${truth}"
echo "dir:   ${dir_name}"

venv_activate=""
if [ -f "${PROJECT_ROOT}/p10_venv/bin/activate" ]; then
    venv_activate="${PROJECT_ROOT}/p10_venv/bin/activate"
elif [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
    venv_activate="${PROJECT_ROOT}/.venv/bin/activate"
else
    echo "Could not find a virtualenv activate script." >&2
    echo "Checked:" >&2
    echo "  ${PROJECT_ROOT}/p10_venv/bin/activate" >&2
    echo "  ${PROJECT_ROOT}/.venv/bin/activate" >&2
    exit 1
fi

cd "${PROJECT_ROOT}" || exit 1

singularity exec --nv \
    -B /ceph/project/tbd/data/:/ceph/project/tbd/data \
    -B "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    /ceph/container/pytorch/pytorch_26.02.sif \
    /bin/bash -lc "set -euo pipefail && \
        cd '${PROJECT_ROOT}' && \
        source '${venv_activate}' && \
        python ./mami/inference.py \
            --model '${model_name}' \
            --data_path '${truth}' \
            --data_type '${type}' \
            --save_path '${dir_name}/data' && \
        npy_count=\$(find '${dir_name}/data' -type f -name '*.npy' | wc -l) && \
        if [ \"\${npy_count}\" -eq 0 ]; then \
            echo 'No prediction .npy files were produced. Check model/data path compatibility.' >&2; \
            exit 1; \
        fi && \
        python ./mami/evaluation.py \
            --pred_path '${dir_name}/data/' \
            --truth_path '${truth}' \
            --type '${type}' \
            --result_path '${dir_name}/results.json' \
            --summary_dir '${dir_name}'"

echo -e "\n--------------------------------------------------------------------------\n"
