# SPDX-License-Identifier: MIT
# Copyright (c) 2025 <Hugin J. Zachariasen, Magnus H. Jensen, Martin C. B. Nielsen, Tobias S. Madsen>.

import argparse
import cv2
import numpy as np
from PIL import Image

import os
import sys

def find_image_files(rgb_path):
    base = rgb_path[:-4]
    suffix = rgb_path[-4:] # TODO: Make dynamic
    g_path = base + "_g" + suffix
    r_path = base + "_r" + suffix
    re_path = base + "_re" + suffix
    nir_path = base + "_nir" + suffix

    return {
        "rgb": rgb_path,
        "g": g_path,
        "r": r_path,
        "re": re_path,
        "nir": nir_path
    }

def load_image_as_array(path: str, resize: bool = False, resize_size: tuple = (512, 480)):
    img = Image.open(path)
    if resize:
        img = np.array(img)
        img = cv2.resize(img, resize_size, interpolation=cv2.INTER_CUBIC)
    return img.astype(np.float32)

def save_array_as_image(array, path):
    img = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
    img.save(path)

def ndvi_error_map(pred, gt):
    pred_nir = load_image_as_array(pred_files["nir"], True, (1280, 720))
    pred_r = load_image_as_array(pred_files["r"], True, (1280, 720))
    gt_nir = load_image_as_array(gt_files["nir"], True, (1280, 720))
    gt_r = load_image_as_array(gt_files["r"], True, (1280, 720))
    
    pred = (pred_nir - pred_r) / (pred_nir + pred_r + 1e-6)
    gt = (gt_nir - gt_r) / (gt_nir + gt_r + 1e-6)            
    error_map = np.abs(pred - gt)
    return error_map

def ndre_error_map(pred, gt):
    pred_nir = load_image_as_array(pred_files["nir"])
    pred_re = load_image_as_array(pred_files["re"])
    gt_nir = load_image_as_array(gt_files["nir"])
    gt_re = load_image_as_array(gt_files["re"])
    
    pred = (pred_nir - pred_re) / (pred_nir + pred_re + 1e-6)
    gt = (gt_nir - gt_re) / (gt_nir + gt_re + 1e-6)            
    error_map = np.abs(pred - gt)
    return error_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate error maps for given prediction and ground truth.")

    # These are the path to the predicted and ground truth of the rgb images.
    # Expects _re, _g, _r, _nir suffix
    parser.add_argument("--pred_path", default="data/pred.JPG")
    parser.add_argument("--gt_path", default="data/ground_truth.TIF")
    parser.add_argument("--error_map_type", default="ndvi", choices=["all", "ndvi", "ndre"])
    args = parser.parse_args()

    pred_files = find_image_files(args.pred_path)
    gt_files = find_image_files(args.gt_path)

    match args.error_map_type:
        case "ndvi":
            ndvi_error_map = ndvi_error_map(pred_files, gt_files)
            save_array_as_image(ndvi_error_map * 255, "ndvi_error_map.png")
        case "ndre":
            ndre_error_map = ndre_error_map(pred_files, gt_files)
            save_array_as_image(ndre_error_map * 255, "ndre_error_map.png")
        case "all":
            # ndvi
            ndvi_error_map = ndvi_error_map(pred_files, gt_files)
            save_array_as_image(ndvi_error_map * 255, "ndvi_error_map.png")
            
            # ndre
            ndre_error_map = ndre_error_map(pred_files, gt_files)
            save_array_as_image(ndre_error_map * 255, "ndre_error_map.png")
    