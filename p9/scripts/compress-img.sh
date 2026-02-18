#!/bin/bash

# Check usage
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 /path/to/image_directory"
    exit 1
fi

IMG_DIR="$1"

if [ ! -d "$IMG_DIR" ]; then
    echo "Error: '$IMG_DIR' is not a directory"
    exit 1
fi

# Compression levels
qualities=(10 20 25 30 40 50 60 70 75 80 90 100)

IMG_DIR="$(realpath "$IMG_DIR")"

# Loop over compression levels
for q in "${qualities[@]}"; do
    outdir="${IMG_DIR}/comp_${q}"
    mkdir -p "$outdir"

    # Process JPG files (recompress)
    for img in "$IMG_DIR"/*.JPG; do
        [ -e "$img" ] || continue
        filename="$(basename "$img")"
        convert "$img" -quality "$q" "$outdir/$filename"
        echo "Compressed JPG -> comp_${q}/$filename"
    done

    # Copy TIF files (no modification)
    for tif in "$IMG_DIR"/*.TIF; do
        [ -e "$tif" ] || continue
        cp "$tif" "$outdir/"
        echo "Copied TIF -> comp_${q}/$(basename "$tif")"
    done
done
