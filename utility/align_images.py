# SPDX-License-Identifier: MIT
# Copyright (c) 2025 <Hugin J. Zachariasen, Magnus H. Jensen, Martin C. B. Nielsen, Tobias S. Madsen>.

import cv2
import numpy as np
import os
import glob

def read_and_preprocess(path, grayscale=True):
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img

def align_image(ref_img, img_to_align, use_affine=False):
    # Always use grayscale for feature detection
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY) if len(ref_img.shape) == 3 else ref_img
    align_gray = img_to_align if len(img_to_align.shape) == 2 else cv2.cvtColor(img_to_align, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(align_gray, None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    if len(matches) < 4:
        raise Exception("Not enough matches found for alignment.")
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    if use_affine:
        M, _ = cv2.estimateAffinePartial2D(pts2, pts1, method=cv2.RANSAC)
        aligned = cv2.warpAffine(img_to_align, M, (ref_img.shape[1], ref_img.shape[0]))
    else:
        H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
        aligned = cv2.warpPerspective(img_to_align, H, (ref_img.shape[1], ref_img.shape[0]))
    return aligned

def crop_to_valid_overlap(images, extra_margin=0.05):
    # Find intersection of non-black (non-zero) regions in all images
    masks = []
    for img in images:
        if len(img.shape) == 3:
            mask = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)
        else:
            mask = (img > 0).astype(np.uint8)
        masks.append(mask)
    overlap = np.logical_and.reduce(masks).astype(np.uint8)
    # Erode to avoid edge black bars
    kernel = np.ones((15, 15), np.uint8)
    overlap = cv2.erode(overlap, kernel, iterations=1)
    if not np.any(overlap):
        raise Exception("No overlapping area found.")
    x, y, w, h = cv2.boundingRect(overlap)
    # Apply extra margin (zoom in)
    margin_x = int(w * extra_margin)
    margin_y = int(h * extra_margin)
    x += margin_x
    y += margin_y
    w -= 2 * margin_x
    h -= 2 * margin_y
    return [img[y:y+h, x:x+w] for img in images]

def find_image_sets(input_folder):
    # Find all unique base names for image sets in the folder
    pattern = os.path.join(input_folder, '*_MS_G.TIF')
    green_files = glob.glob(pattern)
    sets = []
    for green_path in green_files:
        base = green_path.replace('_MS_G.TIF', '')
        paths = {
            'rgb': base.replace('_MS', '') + '_D.JPG',
            'green': base + '_MS_G.TIF',
            'red': base + '_MS_R.TIF',
            'red_edge': base + '_MS_RE.TIF',
            'nir': base + '_MS_NIR.TIF',
        }
        if all(os.path.exists(paths[k]) for k in paths):
            sets.append(paths)
    return sets

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Batch align and crop multispectral images.')
    parser.add_argument('--input_folder', type=str, required=True, help='Input folder containing image sets')
    parser.add_argument('--output_folder', type=str, help='Output folder for aligned/cropped images', default="data/aligned_images")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    image_sets = find_image_sets(args.input_folder)
    print(f'Found {len(image_sets)} image sets.')

    for paths in image_sets:
        try:
            ref_img = read_and_preprocess(paths['green'], grayscale=True)
            images = []
            rgb_img = read_and_preprocess(paths['rgb'], grayscale=False)
            aligned_rgb = align_image(ref_img, rgb_img, use_affine=True)
            images.append(aligned_rgb)
            for band in ['green', 'red', 'red_edge', 'nir']:
                img = read_and_preprocess(paths[band])
                aligned = align_image(ref_img, img, use_affine=True)
                images.append(aligned)
            cropped_images = crop_to_valid_overlap(images)
            # Save results
            base_name = os.path.basename(paths['green']).replace('_MS_G.TIF', '')
            for i, band in enumerate(['_D.JPG', '_MS_G.TIF', '_MS_R.TIF', '_MS_RE.TIF', '_MS_NIR.TIF']):
                out_path = os.path.join(args.output_folder, f'{base_name}_aligned_cropped{band}')
                cv2.imwrite(out_path, cropped_images[i])
            print(f'Processed {base_name}')
        except Exception as e:
            print(f'Failed to process set {paths}: {e}')
