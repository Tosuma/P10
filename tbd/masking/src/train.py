from __future__ import annotations

import argparse
import csv
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.config import dump_config, load_config
from src.datasets import IMAGENET_MEAN, IMAGENET_STD, build_dataloader, compute_channel_stats
from src.losses import BCEDiceLoss
from src.metrics import ConfusionTotals, threshold_predictions
from src.model import build_model
from src.utils import (
    device_from_config,
    ensure_dir,
    get_git_commit,
    move_batch_to_device,
    package_versions,
    read_json,
    set_seed,
    timestamp_utc,
    write_json,
)


def prepare_normalization(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    modality = config["modality"]
    stats_path = Path(config["paths"]["normalization_stats"])
    ensure_dir(stats_path.parent)

    if modality == "rgb":
        stats = {"mean": IMAGENET_MEAN.tolist(), "std": IMAGENET_STD.tolist(), "source": "imagenet"}
        write_json(stats_path, stats)
        return stats

    if stats_path.exists():
        return read_json(stats_path)

    stats = compute_channel_stats(
        sample_manifest_path=config["paths"]["train_manifest"],
        patch_manifest_path=config["paths"]["train_patch_manifest"],
        modality=modality,
    )
    stats["source"] = "train_split_patch_stats"
    write_json(stats_path, stats)
    write_json(run_dir / "normalization_stats.json", stats)
    return stats


def run_epoch(
    model: torch.nn.Module,
    loader,
    criterion,
    device: torch.device,
    threshold: float,
    optimizer=None,
    scaler=None,
    grad_clip: float | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = ConfusionTotals()
    total_loss = 0.0
    total_batches = 0

    if device.type == "cuda":
        amp_context = torch.autocast(device_type="cuda", dtype=torch.float16)
    else:
        amp_context = nullcontext()

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        images = batch["image"]
        masks = batch["mask"].unsqueeze(1)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with amp_context:
            logits = model(images)
            loss = criterion(logits, masks)

        if training:
            assert scaler is not None
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

        preds = threshold_predictions(logits.detach(), threshold)
        totals.update(preds.cpu(), masks.detach().cpu())
        total_loss += loss.detach().item()
        total_batches += 1

    metrics = totals.compute()
    metrics["loss"] = total_loss / max(total_batches, 1)
    return metrics


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, best_val_iou: float, config: dict[str, Any]) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,
            "best_val_iou": best_val_iou,
            "config": config,
        },
        path,
    )


def maybe_freeze_encoder(model: torch.nn.Module, freeze: bool) -> None:
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return
    for parameter in encoder.parameters():
        parameter.requires_grad = not freeze


def main() -> None:
    parser = argparse.ArgumentParser(description="Train binary weed segmentation model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    set_seed(int(config["seed"]))

    run_name = f"{config['experiment_name']}_seed{config['seed']}_{timestamp_utc()}"
    run_dir = ensure_dir(Path(config["paths"]["run_root"]) / run_name)
    checkpoint_dir = ensure_dir(run_dir / "checkpoints")
    log_dir = ensure_dir(run_dir / "logs")
    metrics_dir = ensure_dir(run_dir / "metrics")

    dump_config(run_dir / "config.yaml", config)
    metadata = {
        "git_commit": get_git_commit(),
        "seed": config["seed"],
        "packages": package_versions(["torch", "numpy", "PIL"]),
    }
    write_json(run_dir / "run_metadata.json", metadata)

    normalization = prepare_normalization(config, run_dir)
    device = device_from_config(config.get("device"))

    train_loader = build_dataloader(
        sample_manifest_path=config["paths"]["train_manifest"],
        patch_manifest_path=config["paths"]["train_patch_manifest"],
        modality=config["modality"],
        normalization=normalization,
        batch_size=int(config["training"]["batch_size"]),
        num_workers=int(config["training"]["num_workers"]),
        is_train=True,
        transform_config=config,
        seed=int(config["seed"]),
    )
    val_loader = build_dataloader(
        sample_manifest_path=config["paths"]["val_manifest"],
        patch_manifest_path=config["paths"]["val_patch_manifest"],
        modality=config["modality"],
        normalization=normalization,
        batch_size=int(config["training"]["batch_size"]),
        num_workers=int(config["training"]["num_workers"]),
        is_train=False,
        transform_config=config,
        seed=int(config["seed"]),
    )

    model = build_model(int(config["in_channels"])).to(device)
    criterion = BCEDiceLoss(
        bce_weight=float(config["loss"]["bce_weight"]),
        dice_weight=float(config["loss"]["dice_weight"]),
    )
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(config["training"]["scheduler"]["factor"]),
        patience=int(config["training"]["scheduler"]["patience"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    freeze_epochs = int(config["training"].get("freeze_encoder_epochs", 0))
    best_val_iou = -1.0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    threshold = float(config["evaluation"]["threshold"])
    early_stopping_patience = int(config["training"]["early_stopping_patience"])

    csv_path = log_dir / "history.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_iou",
                "train_dice",
                "val_loss",
                "val_iou",
                "val_dice",
                "val_precision",
                "val_recall",
                "lr",
            ],
        )
        writer.writeheader()

        for epoch in range(1, int(config["training"]["epochs"]) + 1):
            maybe_freeze_encoder(model, freeze=epoch <= freeze_epochs)
            train_metrics = run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                device=device,
                threshold=threshold,
                optimizer=optimizer,
                scaler=scaler,
                grad_clip=float(config["training"]["grad_clip"]),
            )
            with torch.no_grad():
                val_metrics = run_epoch(
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                    device=device,
                    threshold=threshold,
                )

            scheduler.step(val_metrics["iou"])
            current_lr = optimizer.param_groups[0]["lr"]
            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_iou": train_metrics["iou"],
                "train_dice": train_metrics["dice"],
                "val_loss": val_metrics["loss"],
                "val_iou": val_metrics["iou"],
                "val_dice": val_metrics["dice"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "lr": current_lr,
            }
            writer.writerow(row)
            handle.flush()
            history.append(row)

            save_checkpoint(checkpoint_dir / "latest.pt", model, optimizer, scheduler, epoch, best_val_iou, config)
            if val_metrics["iou"] > best_val_iou:
                best_val_iou = val_metrics["iou"]
                epochs_without_improvement = 0
                save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, scheduler, epoch, best_val_iou, config)
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= early_stopping_patience:
                break

    write_json(metrics_dir / "history.json", history)
    write_json(metrics_dir / "best_summary.json", {"best_val_iou": best_val_iou, "run_dir": str(run_dir.resolve())})
    print(str(run_dir.resolve()))


if __name__ == "__main__":
    main()
