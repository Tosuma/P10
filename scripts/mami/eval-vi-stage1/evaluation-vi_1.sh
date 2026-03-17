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
  local model=""
  local prediction_path=""
  local truth_path=""
  local data_type=""
  local result_path=""
  local print_results=false
  local save_images=true
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
    python ./mami/evaluation.py
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

sri_path="./data/sri-lanka-aligned/"
weedy_path="./data/WeedyRice/"

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


for ndre in $(seq -f "%.1f" 0.0 0.1 0.1); do
  for ndvi in $(seq -f "%.1f" 0.0 0.1 1.0); do
    run_eval \
      --model "./checkpoints/vi/finals/vi-kaz-re_${ndre}-vi_${ndvi}_stage1_best.pth" \
      --pred "results/vi/re_${ndre}_vi_${ndvi}_stage1---weedy-rice/data/" \
      --truth "${weedy_path}" \
      --type "Weedy-Rice" \
      --out "results/vi/re_${ndre}_vi_${ndvi}_stage1---weedy-rice/results.json" \
      --save-images
  done
done