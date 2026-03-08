# SPDX-License-Identifier: MIT
# Copyright (c) 2025 <Hugin J. Zachariasen, Magnus H. Jensen, Martin C. B. Nielsen, Tobias S. Madsen>.

import argparse
import os
import cv2
import numpy as np
from collections import defaultdict
import math

BAND_SUFFIXES = ["_MS_G.TIF", "_MS_R.TIF", "_MS_RE.TIF", "_MS_NIR.TIF"]

def patch_images(rgb_path, patch_size=256):
    """
    Loads each of the bands and resizes the smallest size of them.
    Then it generates the patches and saves to disk.
    """

    # Load image and bands
    print(rgb_path)
    rgb = rgb_path
    bands_paths = [rgb]
    band_path = rgb_path.replace("_D.JPG", "")
    for x in {band_path + suffix for suffix in BAND_SUFFIXES}:
        bands_paths.append(x)

    # Find the smallest size image
    smallest_width = math.inf
    smallest_height = math.inf

    for band in bands_paths:
        img = cv2.imread(band)
        if img is None:
            print(f"[WARN] Cannot read image: {band}. Skipping this band.")
            continue
        h, w, = img.shape[:2]
        if h < smallest_height: smallest_height = h
        if w < smallest_width: smallest_width = w

    bands = []
    # Read each band path as image
    for band_path in bands_paths:
        img = cv2.imread(band_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[WARN] Cannot read image: {band_path}. Skipping this band.")
            continue
        bands.append(img)

    # Resize images
    for i, x in enumerate(bands):
        bands[i] = cv2.resize(x, (smallest_width, smallest_height), interpolation=cv2.INTER_CUBIC)

    # Calculate number of patches (small overlap is allowed)
    cols = smallest_width // patch_size
    rows = smallest_height // patch_size

    print(cols,"cols", rows, "rows")

    # Create patches and save to disk
    patches = []
    for idx, band in enumerate(bands):
        counter = 0
        for i in range(rows):
            for j in range(cols):
                counter += 1
                y0 = i * (smallest_height // rows)
                x0 = j * (smallest_width // cols)
                patch = band[y0:y0 + patch_size, x0:x0 + patch_size]
                patches.append(patch)

                # Save to disk
                patch_path = f"{bands_paths[idx][:-4]}_{counter}{bands_paths[idx][-4:]}"
                cv2.imwrite(patch_path, patch)
                print(patch_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creates patches from spectral bands.")
    parser.add_argument("--data_path", default="data")
    parser.add_argument("--patch_size", default=256, type=int)
    args = parser.parse_args()

    root_dir = args.data_path
    for image_name in os.listdir(root_dir):
        # Find the JPG rgb files in the directory 
        # Also filters weird singletons in the dataset
        if image_name.endswith("_D.JPG"):
            rgb_path = os.path.join(root_dir, image_name)
            patch_images(rgb_path)

    print(f"\nDone creating the dataset!")
