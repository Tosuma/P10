#!/bin/env bash

# Usage: ./run_all.sh /path/to/json_dir
DIR="${1:-}"

if [[ -z "$DIR" ]]; then
  echo "Usage: $0 /path/to/json_dir"
  exit 1
fi

if [[ ! -d "$DIR" ]]; then
  echo "Error: '$DIR' is not a directory"
  exit 1
fi

OUT_DIR="./results/comp"
mkdir -p "$OUT_DIR"

shopt -s nullglob

for file_path in "$DIR"/*.json; do
  file_name="$(basename "$file_path")"

  # Match: [name]-c[number].json
  # Example: experimentA-c12.json  -> name=experimentA, number=12
  if [[ "$file_name" =~ ^(.+)-c([0-9]+)\.json$ ]]; then
    name="${BASH_REMATCH[1]}"
    number="${BASH_REMATCH[2]}"

    excel_path="${OUT_DIR}/${name}-comp-${number}.xlsx"

    echo "Processing: $file_name -> $excel_path"
    python ./utils/results_manager.py --json-path "$file_path" --excel-path "$excel_path"
  else
    echo "Skipping (pattern mismatch): $file_name"
  fi
done
