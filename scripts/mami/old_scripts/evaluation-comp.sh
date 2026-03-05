#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 <Hugin J. Zachariasen, Magnus H. Jensen, Martin C. B. Nielsen, Tobias S. Madsen>.


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
  local prediction_path=""
  local truth_path=""
  local data_type=""
  local model=""
  local result_path=""
  local print_results=false
  local save_images=false
  local single_image=false
  local jpg_image=""

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
  local eval_cmd=(
    python ./mami/inference.py
    --model "$model"
    --data_path "$truth_path"
    --data_type "$data_type"
    --save_path "$prediction_path"
  )

  # Conditionally add flag
  if [[ "$save_images" == true ]]; then
    eval_cmd+=(--save_images)
  fi

  if [[ "$single_image" == true ]]; then
    eval_cmd+=(--single)
    eval_cmd+=(--jpg "$jpg_image")
  fi

  "${eval_cmd[@]}"

  echo "=== Beginning evaluation ==="
  local vali_cmd=(
    python ./mami/validate.py
    --pred_path "$prediction_path"
    --truth_path "$truth_path"
    --type "$data_type"
    --result_path "$result_path"
  )

  if [[ "$print_results" == true ]]; then
    vali_cmd+=(--print_results)
  fi

  "${vali_cmd[@]}"
  echo -e "\n--------------------------------------------------------------------------\n"
}

# Model paths
WEEDY_MODEL="./checkpoints/300/tl-weedy-rice/stage3_best_final.pth"
SRILANKA_MODEL="./checkpoints/300/tl-sri-lanka/stage3_best_final.pth"

# Dataset types
WEEDY_TYPE="Weedy-Rice"
SRILANKA_TYPE="Sri-Lanka"

# Base directories
WEEDY_BASE="./data/WeedyRice-comp"
SRILANKA_BASE="./data/Sri-Lanka-comp"

# Results base
RESULTS_BASE="results/comp"

# ============================================================
# Helper: run evaluations for a base directory
# ============================================================
run_dataset() {
    local base_dir="$1"
    local model="$2"
    local data_type="$3"


    echo "============================================================"
    echo "Running dataset: $base_dir"
    echo "============================================================"

    for comp_dir in "$base_dir"/comp_*; do
        [ -d "$comp_dir" ] || continue

        comp_name="$(basename "$comp_dir")"
        comp_level="${comp_name#comp_}"

        pred_dir="${RESULTS_BASE}/tl-stage3---${comp_level}---${data_type}/data/"
        out_file="${RESULTS_BASE}/${data_type}-c${comp_level}.json"

        echo "------------------------------------------------------------"
        echo "Compression level: $comp_level"
        echo "Truth dir       : $comp_dir"
        echo "Pred dir        : $pred_dir"
        echo "Out dir         : $out_file"
        echo "------------------------------------------------------------"

        run_eval \
            --model "$model" \
            --pred "$pred_dir" \
            --truth "$comp_dir" \
            --type "$data_type" \
            --out "$out_file" \
            --save-images
    done
}

# ============================================================
# RUN DATASETS SEQUENTIALLY (NOT IN A SINGLE LOOP)
# ============================================================

# 1. WeedyRice
run_dataset \
    "$WEEDY_BASE" \
    "$SRILANKA_MODEL" \
    "$WEEDY_TYPE"

# 2. Sri-Lanka
run_dataset \
    "$SRILANKA_BASE" \
    "$WEEDY_MODEL" \
    "$SRILANKA_TYPE"

echo "All evaluations completed."
