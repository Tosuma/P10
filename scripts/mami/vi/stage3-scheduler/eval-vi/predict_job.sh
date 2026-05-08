#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 <Hugin J. Zachariasen, Magnus H. Jensen, Tobias S. Madsen>.

#SBATCH --job-name=pred_mami
#SBATCH --output=logs/inference/pred_mami_%j.out
#SBATCH --error=logs/inference/pred_mami_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00


# run_eval
#  --model         : Model path to test
#  --pred          : The directory where the predictions are placed (the directory will automatically be created)
#  --truth         : The root directory where the ground truth files are located (the dataset)
#  --type          : The type of the dataset
#  --out           : The results out dir (will automatically be created)
#  --print-results : Print the results of an evaluation foreach of the predictions
#  --save-images   : Bool flag to save images
#  --single-image  : Bool flag to run single image
#  --jpg           : Name of the JPG file - not full path

run_eval() {
  local model=""
  local prediction_path=""
  local truth_path=""
  local data_type=""
  local result_path=""
  local print_results=false
  local save_images=true
  local single_image=false
  local jpg_image=""
  local summary_dir=""

  # Parse args
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -p|--pred)
        [[ $# -ge 2 ]] || { echo "Error: $1 requires a value" >&2; return 2; }
        prediction_path="$2"; shift 2 ;;
      -gt|--truth)
        [[ $# -ge 2 ]] || { echo "Error: $1 requires a value" >&2; return 2; }
        truth_path="$2"; shift 2 ;;
      -ty|--type)
        [[ $# -ge 2 ]] || { echo "Error: $1 requires a value" >&2; return 2; }
        data_type="$2"; shift 2 ;;
      -m|--model)
        [[ $# -ge 2 ]] || { echo "Error: $1 requires a value" >&2; return 2; }
        model="$2"; shift 2 ;;
      -o|--out)
        [[ $# -ge 2 ]] || { echo "Error: $1 requires a value" >&2; return 2; }
        result_path="$2"; shift 2 ;;
      --summary)
        [[ $# -ge 2 ]] || { echo "Error: $1 requires a value" >&2; return 2; }
        summary_dir="$2"; shift 2 ;;
      --print-results)
        print_results=true; shift 1;;
      -s|--save-images)
        save_images=true; shift 1;;
      --single-image)
        single_image=true; shift  1;;
      --jpg)
        jpg_image="$2"; shift 2;;
      *)
        echo "Unknown argument: $1" >&2
        echo "Run: run_eval --help" >&2
        return 2
        ;;
    esac
  done

  # Validate required args
  if [[ -z "$prediction_path" || -z "$truth_path" || -z "$data_type" || -z "$model" || -z "$result_path" ]]; then
    echo "Error: Missing required arguments." >&2
    echo "Usage: run_eval -p <prediction_path> -t <truth_path> -d <data_type> -m <model_path> -o <result_path> [--save-images]" >&2
    return 2
  fi

  if [[ "$single_image" == true && -z "$jpg_image" ]]; then
    echo "Error: Missing JPG image name when using 'single-image' flag" >&2
    return 2
  fi


  echo "=== Beginning predictions ==="

  singularity exec --nv \
    /ceph/container/pytorch/pytorch_26.02.sif \
    /bin/bash -lc "source p10_venv/bin/activate && \
        python ./mami/inference.py \
            --model ${model} \
            --data_path ${truth_path} \
            --data_type ${data_type} \
            --save_path ${prediction_path}"
}

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

# Basic required-arg checks (optional but recommended)
: "${dir_name:?Missing --dir_name}"
: "${model_name:?Missing --model_name}"
: "${type:?Missing --loss_ndre_w}"
: "${truth:?Missing --loss_ndvi_w}"

run_eval \
    --model "${model_name}" \
    --pred "${dir_name}/data" \
    --truth "${truth}" \
    --type "${type}" \
    --out "${dir_name}/results.json" \
    --summary "${dir_name}"
    # --save-images
