"""
Evaluation script for trained weedy rice segmentation models.

Loads a saved model checkpoint and evaluates on the test split.

Usage:
    python evaluate.py --data_root ./data/WeedyRice-RGBMS-DB --input_type rgb --seed 42 --model_path ./results/rgb_seed42/best_model.pth
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataset import WeedyRiceDataset, get_transforms
from model import build_unet
from metrics import SegmentationMetrics
from train import FilteredDataset, get_data_dir, seed_everything


def get_config():
    parser = argparse.ArgumentParser(description="Evaluate a trained segmentation model")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--input_type", type=str, required=True,
                        choices=["rgb", "multispectral", "synthetic"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to best_model.pth checkpoint")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--ms_channels", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save results (defaults to model directory)")
    return parser.parse_args()


def main():
    cfg = get_config()
    seed_everything(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Input type: {cfg.input_type}, Seed: {cfg.seed}")
    print(f"Model: {cfg.model_path}")

    # Determine input channels
    in_channels = 3 if cfg.input_type == "rgb" else cfg.ms_channels

    # Recreate the same test split
    mask_dir = os.path.join(cfg.data_root, "Masks")
    all_files = sorted([
        f for f in os.listdir(mask_dir)
        if f.lower().endswith((".png", ".tif", ".tiff"))
    ])
    train_files, temp_files = train_test_split(
        all_files, test_size=0.30, random_state=cfg.seed
    )
    val_files, test_files = train_test_split(
        temp_files, test_size=0.50, random_state=cfg.seed
    )
    print(f"Test split: {len(test_files)} images")

    # Build test dataloader
    image_dir = get_data_dir(cfg.data_root, cfg.input_type)
    mask_dir = os.path.join(cfg.data_root, "Masks")
    test_ds = FilteredDataset(
        file_list=test_files,
        image_dir=image_dir,
        mask_dir=mask_dir,
        input_type=cfg.input_type,
        transform=get_transforms("test", cfg.image_size),
        num_classes=cfg.num_classes,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # Load model
    model = build_unet(in_channels=in_channels, num_classes=cfg.num_classes).to(device)
    model.load_state_dict(torch.load(cfg.model_path, weights_only=True, map_location=device))
    print(f"Model loaded from {cfg.model_path}")

    # Evaluate
    criterion = nn.CrossEntropyLoss()
    metrics = SegmentationMetrics(cfg.num_classes)
    model.eval()
    total_loss = 0.0
    metrics.reset()

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)
            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            metrics.update(preds, masks)

    test_loss = total_loss / len(test_loader.dataset)
    test_results = metrics.compute()

    # Print results
    print("\n" + "=" * 60)
    print("TEST EVALUATION RESULTS")
    print("=" * 60)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test mIoU: {test_results['miou']:.4f}")
    for cls_name, iou in test_results["per_class_iou"].items():
        print(f"  Class {cls_name} IoU: {iou:.4f}")

    # Save results
    output_dir = cfg.output_dir or os.path.dirname(cfg.model_path)
    os.makedirs(output_dir, exist_ok=True)
    results = {
        "input_type": cfg.input_type,
        "seed": cfg.seed,
        "model_path": cfg.model_path,
        "test_loss": test_loss,
        "test_miou": test_results["miou"],
        "test_per_class_iou": test_results["per_class_iou"],
        "confusion_matrix": test_results["confusion_matrix"],
    }
    results_path = os.path.join(output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()