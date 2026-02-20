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
  if [[ -z "$prediction_path" || -z "$truth_path" || -z "$data_type" || -z "$model" || -z "$result_path" ]] then
    echo "Error: Missing required arguments." >&2
    echo "Usage: run_eval -p <prediction_path> -t <truth_path> -d <data_type> -m <model_path> -o <result_path> [--save-images]" >&2
    return 2
  fi

  if [[ "$single_image" == true && -z "$jpg_image" ]] then
    echo "Error: Missing JPG image name when using 'single-image' flag" >&2
    return 2
  fi


  # echo "=== Beginning predictions ==="
  local eval_cmd=(
    python ./mami/inference.py
    --model "$model"
    --data_path "$truth_path"
    --data_type "$data_type"
    --save_path "$prediction_path"
  )

  # Conditionally add flag
  if [[ "$save_images" == true ]] then
    eval_cmd+=(--save_images)
  fi

  if [[ "$single_image" == true ]] then
    eval_cmd+=(--single)
    eval_cmd+=(--jpg "$jpg_image")
  fi

  # "${eval_cmd[@]}"

  echo "=== Beginning evaluation ==="
  local vali_cmd=(
    python ./mami/evaluation.py
    --pred_path "$prediction_path"
    --truth_path "$truth_path"
    --type "$data_type"
    --result_path "$result_path"
  )

  if [[ "$print_results" == true ]] then
    vali_cmd+=(--print_results)
  fi

  "${vali_cmd[@]}"
  echo -e "\n--------------------------------------------------------------------------\n"
}

sri_path="./data/sri-lanka-aligned/"
weedy_path="./data/WeedyRice/"

echo "---------- 1st run ----------"
# Base model stage 1 (trained on Kazakhstan) test on Sri Lanka
run_eval \
  --model "./checkpoints/basemodel-tl-Weed/stage1_best_final.pth" \
  --pred "results/basemodel-stage1---Sri-Lanka/data/" \
  --truth "$sri_path" \
  --type "Sri-Lanka" \
  --out "results/basemodel-stage1---Sri-Lanka/results.json" \
  --save-images

# Base model stage 1 (trained on Kazakhstan) test on Weedy Rice
run_eval \
  --model "./checkpoints/basemodel-tl-Weed/stage1_best_final.pth" \
  --pred "results/basemodel-stage1---Weedy-Rice/data/" \
  --truth "$weedy_path" \
  --type "Weedy-Rice" \
  --out "results/basemodel-stage1---Weedy-Rice/results.json" \
  --save-images

# Base model stage 2 (trained on Kazakhstan + Weedy-Rice) test on Sri-Lanka
run_eval \
  --model "./checkpoints/100/tl-weedy-rice/stage2_best_final.pth" \
  --pred "results/100/tl-weedy-rice-stage2---sri-lanka/data/" \
  --truth "$sri_path" \
  --type "Sri-Lanka" \
  --out "results/100/tl-weedy-rice-stage2---sri-lanka/results.json" \
  --save-images

# Base model stage 3 (trained on Kazakhstan + Weedy-Rice) test on Sri-Lanka
run_eval \
  --model "./checkpoints/100/tl-weedy-rice/stage3_best_final.pth" \
  --pred "results/100/tl-weedy-rice-stage3---sri-lanka/data/" \
  --truth "$sri_path" \
  --type "Sri-Lanka" \
  --out "results/100/tl-weedy-rice-stage3---sri-lanka/results.json" \
  --save-images

echo "---------- 2nd run ----------"
# Base model stage 2 (trained on Kazakhstan + Sri-Lanka) test on Weedy-Rice
run_eval \
  --model "./checkpoints/100/tl-sri-lanka/stage2_best_final.pth" \
  --pred "results/100/tl-sri-lanka-stage-2---weedy-rice/data/" \
  --truth "$weedy_path" \
  --type "Weedy-Rice" \
  --out "results/100/tl-sri-lanka-stage-2---weedy-rice/results.json" \
  --save-images

# Base model stage 3 (trained on Kazakhstan + Sri-Lanka) test on Weedy-Rice
run_eval \
  --model "./checkpoints/100/tl-sri-lanka/stage3_best_final.pth" \
  --pred "results/100/tl-sri-lanka-stage-3---weedy-rice/data/" \
  --truth "$weedy_path" \
  --type "Weedy-Rice" \
  --out "results/100/tl-sri-lanka-stage-3---weedy-rice/results.json" \
  --save-images

echo "---------- 3rd run ----------"
# Base model stage 3 (trained on Kazakhstan + Sri-Lanka) test on Weedy-Rice
run_eval \
  --model "./checkpoints/100/sri-lanka-stage3-only/stage3_best_final.pth" \
  --pred "results/100/sri-lanka-stage3-only---weedy-rice/data/" \
  --truth "$weedy_path" \
  --type "Weedy-Rice" \
  --out "results/100/sri-lanka-stage3-only---weedy-rice/results.json" \
  --save-images

echo "---------- 4th run ----------"
# Base model stage 3 (trained on Kazakhstan + Weedy-Rice) test on Sri-Lanka
run_eval \
  --model "./checkpoints/100/weed-rice-stage3-only/stage3_best_final.pth" \
  --pred "results/100/weed-rice-stage3-only---sri-lanka/data/" \
  --truth "$sri_path" \
  --type "Sri-Lanka" \
  --out "results/100/weed-rice-stage3-only---sri-lanka/results.json" \
  --save-images

echo "---------- 5th run ----------"
# Base model stage 2 (trained on Kazakhstan + Sri-Lanka) test on Weedy-Rice
run_eval \
  --model "./checkpoints/100/sri-lanka-stage2-trained-on-stage3/stage2_best_final.pth" \
  --pred "results/100/Sri-lanka-stage2-trained-on-stage3---weedy-rice/data/" \
  --truth "$weedy_path" \
  --type "Weedy-Rice" \
  --out "results/100/Sri-lanka-stage2-trained-on-stage3---weedy-rice/results.json" \
  --save-images

echo "---------- 6th run ----------"
# Base model stage 2 (trained on Kazakhstan + Weedy-Rice) test on Sri-Lanka
run_eval \
  --model "./checkpoints/100/weedy-rice-stage2-trained-on-stage3/stage2_best_final.pth" \
  --pred "results/100/Weedy-Rice-stage2-trained-on-stage3---sri-lanka/data/" \
  --truth "$sri_path" \
  --type "Sri-Lanka" \
  --out "results/100/Weedy-Rice-stage2-trained-on-stage3---sri-lanka/results.json" \
  --save-images
