"""
Training script for the weedy rice segmentation experiment.

Runs one variant at a time. Use run_experiment.py to run all three
variants across multiple seeds.

Usage:
    python train.py --input_type rgb --seed 42 --data_root ./data/WeedyRice-RGBMS-DB
    python train.py --input_type multispectral --seed 42 --data_root ./data/WeedyRice-RGBMS-DB
    python train.py --input_type synthetic --seed 42 --data_root ./data/WeedyRice-RGBMS-DB
"""

import argparse
import os
import random
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataset import WeedyRiceDataset, get_transforms
from model import build_unet
from metrics import SegmentationMetrics


# ──────────────────────────── Configuration ────────────────────────────

def get_config():
    parser = argparse.ArgumentParser(description="Weedy Rice Segmentation Experiment")

    # Data
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing RGB/, Multispectral/, Synthetic/, Masks/")
    parser.add_argument("--input_type", type=str, required=True,
                        choices=["rgb", "multispectral", "synthetic"],
                        help="Input data variant to train on")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Input image size (square crop)")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of segmentation classes")

    # Multispectral config
    parser.add_argument("--ms_channels", type=int, default=4,
                        help="Number of multispectral channels (G, R, RE, NIR)")

    # Training
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay for AdamW")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (epochs)")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Output
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save results and checkpoints")

    return parser.parse_args()


# ──────────────────────────── Seed everything ──────────────────────────

def seed_everything(seed: int):
    """Ensure reproducibility across runs with the same seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────── Data loading ─────────────────────────────

def get_data_dir(data_root: str, input_type: str) -> str:
    """Map input type to the correct image directory."""
    mapping = {
        "rgb": os.path.join(data_root, "RGB"),
        "multispectral": os.path.join(data_root, "Multispectral"),
        "synthetic": os.path.join(data_root, "Synthetic"),
    }
    return mapping[input_type]


def create_splits(data_root: str, seed: int):
    """
    Create train/val/test splits from mask filenames.

    CRITICAL: The same seed produces the same splits across all three
    variants. This ensures you're comparing data types on identical
    image sets.

    Returns lists of mask filenames for each split.
    """
    mask_dir = os.path.join(data_root, "Masks")
    all_files = sorted([
        f for f in os.listdir(mask_dir)
        if f.lower().endswith((".png", ".tif", ".tiff"))
    ])

    # 70/15/15 split
    train_files, temp_files = train_test_split(
        all_files, test_size=0.30, random_state=seed
    )
    val_files, test_files = train_test_split(
        temp_files, test_size=0.50, random_state=seed
    )

    print(f"Split sizes — Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    return train_files, val_files, test_files


class FilteredDataset(WeedyRiceDataset):
    """WeedyRiceDataset filtered to a subset of files."""
    def __init__(self, file_list, **kwargs):
        super().__init__(**kwargs)
        # file_list contains mask filenames like "stem.png"
        stems_in_split = {os.path.splitext(f)[0] for f in file_list}
        self.image_stems = [s for s in self.image_stems if s in stems_in_split]


def build_dataloaders(cfg, train_files, val_files, test_files):
    """Build DataLoaders for each split."""
    image_dir = get_data_dir(cfg.data_root, cfg.input_type)
    mask_dir = os.path.join(cfg.data_root, "Masks")

    train_ds = FilteredDataset(
        file_list=train_files,
        image_dir=image_dir,
        mask_dir=mask_dir,
        input_type=cfg.input_type,
        transform=get_transforms("train", cfg.image_size),
        num_classes=cfg.num_classes,
    )
    val_ds = FilteredDataset(
        file_list=val_files,
        image_dir=image_dir,
        mask_dir=mask_dir,
        input_type=cfg.input_type,
        transform=get_transforms("val", cfg.image_size),
        num_classes=cfg.num_classes,
    )
    test_ds = FilteredDataset(
        file_list=test_files,
        image_dir=image_dir,
        mask_dir=mask_dir,
        input_type=cfg.input_type,
        transform=get_transforms("test", cfg.image_size),
        num_classes=cfg.num_classes,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# ──────────────────────────── Training loop ────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, metrics):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    metrics.reset()

    for images, masks in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        metrics.update(preds, masks)

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, metrics.compute()


@torch.no_grad()
def evaluate(model, loader, criterion, device, metrics):
    """Evaluate model. Returns average loss and metrics."""
    model.eval()
    total_loss = 0.0
    metrics.reset()

    for images, masks in tqdm(loader, desc="Eval", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        metrics.update(preds, masks)

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, metrics.compute()


# ──────────────────────────── Main ─────────────────────────────────────

def main():
    cfg = get_config()
    seed_everything(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Input type: {cfg.input_type}, Seed: {cfg.seed}")

    # Determine input channels
    if cfg.input_type == "rgb":
        in_channels = 3
    else:
        in_channels = cfg.ms_channels

    # Create output directory
    run_name = f"{cfg.input_type}_seed{cfg.seed}"
    run_dir = os.path.join(cfg.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Data
    train_files, val_files, test_files = create_splits(cfg.data_root, cfg.seed)
    train_loader, val_loader, test_loader = build_dataloaders(
        cfg, train_files, val_files, test_files
    )

    # Model
    model = build_unet(
        in_channels=in_channels,
        num_classes=cfg.num_classes,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=1e-6
    )

    # Metrics
    train_metrics = SegmentationMetrics(cfg.num_classes)
    val_metrics = SegmentationMetrics(cfg.num_classes)
    test_metrics = SegmentationMetrics(cfg.num_classes)

    # Training loop with early stopping
    best_val_miou = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_miou": [], "val_miou": []}

    for epoch in range(1, cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.epochs}")

        train_loss, train_results = train_one_epoch(
            model, train_loader, criterion, optimizer, device, train_metrics
        )
        val_loss, val_results = evaluate(
            model, val_loader, criterion, device, val_metrics
        )
        scheduler.step()

        # Log
        print(f"  Train Loss: {train_loss:.4f} | Train mIoU: {train_results['miou']:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val mIoU:   {val_results['miou']:.4f}")
        for cls_name, iou in val_results["per_class_iou"].items():
            print(f"    Class {cls_name} IoU: {iou:.4f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_miou"].append(train_results["miou"])
        history["val_miou"].append(val_results["miou"])

        # Early stopping check
        if val_results["miou"] > best_val_miou:
            best_val_miou = val_results["miou"]
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))
            print(f"  -> New best model saved (mIoU: {best_val_miou:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"  -> Early stopping at epoch {epoch}")
                break

    # ──────────── Final test evaluation ────────────
    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)

    model.load_state_dict(torch.load(os.path.join(run_dir, "best_model.pth"), weights_only=True))
    test_loss, test_results = evaluate(
        model, test_loader, criterion, device, test_metrics
    )

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test mIoU: {test_results['miou']:.4f}")
    for cls_name, iou in test_results["per_class_iou"].items():
        print(f"  Class {cls_name} IoU: {iou:.4f}")

    # Save results
    results = {
        "input_type": cfg.input_type,
        "seed": cfg.seed,
        "best_val_miou": best_val_miou,
        "test_loss": test_loss,
        "test_miou": test_results["miou"],
        "test_per_class_iou": test_results["per_class_iou"],
        "epochs_trained": len(history["train_loss"]),
        "model_params": param_count,
        "config": vars(cfg),
        "timestamp": datetime.now().isoformat(),
    }

    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save training history
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nResults saved to {run_dir}")
    return results


if __name__ == "__main__":
    main()