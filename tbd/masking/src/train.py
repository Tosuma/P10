from __future__ import annotations

import argparse
import csv
import os
from copy import deepcopy
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.config import dump_config, load_config
from src.datasets import IMAGENET_MEAN, IMAGENET_STD, build_dataloader, compute_channel_stats
from src.evaluate import build_overall_metrics_payload, evaluate_loader, load_checkpoint, reconstruct_image_metrics
from src.losses import BCEDiceLoss
from src.metrics import ConfusionTotals, threshold_predictions
from src.model import build_model, resolve_model_config
from src.targets import resolve_target_config
from src.utils import (
    device_from_config,
    ensure_dir,
    get_git_commit,
    move_batch_to_device,
    package_versions,
    read_csv_rows,
    read_json,
    resolve_seed,
    set_seed,
    setup_file_logger,
    timestamp_utc,
    write_json,
)
from src.visualize import save_binary_mask


HISTORY_FIELDNAMES = [
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
]


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


def evaluate_test_split(
    checkpoint_path: Path,
    config: dict[str, Any],
    normalization: dict[str, Any],
    device: torch.device,
    threshold: float,
) -> dict[str, Any]:
    target_config = resolve_target_config(config)
    model, _, _ = load_checkpoint(str(checkpoint_path), device)
    test_loader = build_dataloader(
        sample_manifest_path=config["paths"]["test_manifest"],
        patch_manifest_path=config["paths"]["test_patch_manifest"],
        modality=config["modality"],
        normalization=normalization,
        batch_size=int(config["training"]["batch_size"]),
        num_workers=int(config["training"]["num_workers"]),
        is_train=False,
        transform_config=config,
        seed=config["seed"],
    )

    results = evaluate_loader(model, test_loader, device, threshold=threshold, config=config)
    sample_rows = read_csv_rows(config["paths"]["test_manifest"])
    image_results = reconstruct_image_metrics(sample_rows, results["reconstruction_payload"], threshold, config=config)
    mask_dir = ensure_dir(checkpoint_path.parents[1] / "metrics" / "test_masks")
    for visual in image_results["visuals"]:
        save_binary_mask(mask_dir / f"{visual['sample_id']}.png", visual["pred_mask"])

    summary, primary_view = build_overall_metrics_payload(results, image_results)
    summary.update(
        {
        "checkpoint": str(checkpoint_path.resolve()),
        "split": "test",
        "target_mode": target_config["mode"],
        "mask_dir": str(mask_dir.resolve()),
        "num_patches": len(results["views"][primary_view]["per_patch_rows"]),
        "num_images": len(image_results["views"][primary_view]),
        }
    )
    return summary


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_val_iou: float,
    best_epoch: int,
    best_metrics: dict[str, Any] | None,
    epochs_without_improvement: int,
    config: dict[str, Any],
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tmp_path = path.with_name(f".{path.name}.tmp")
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "best_val_iou": best_val_iou,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "epochs_without_improvement": epochs_without_improvement,
        "config": config,
    }
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def resolve_resume_checkpoint(resume_checkpoint: str | None, resume_run_dir: str | None) -> Path | None:
    if resume_checkpoint and resume_run_dir:
        raise ValueError("Use either --resume-checkpoint or --resume-run-dir, not both.")
    if resume_run_dir:
        return Path(resume_run_dir) / "checkpoints" / "latest.pt"
    if resume_checkpoint:
        return Path(resume_checkpoint)
    return None


def run_dir_from_checkpoint(checkpoint_path: Path) -> Path:
    if checkpoint_path.parent.name != "checkpoints":
        raise ValueError("Resume checkpoints must live under <run_dir>/checkpoints/.")
    return checkpoint_path.parent.parent


def coerce_history_row(row: dict[str, Any]) -> dict[str, float | int]:
    coerced: dict[str, float | int] = {"epoch": int(float(row["epoch"]))}
    for field in HISTORY_FIELDNAMES:
        if field == "epoch":
            continue
        coerced[field] = float(row[field])
    return coerced


def read_history_rows(csv_path: Path, json_path: Path, max_epoch: int | None = None) -> list[dict[str, float | int]]:
    raw_rows: list[dict[str, Any]]
    if csv_path.is_file():
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            raw_rows = list(csv.DictReader(handle))
    elif json_path.is_file():
        raw_rows = read_json(json_path)
    else:
        raw_rows = []

    rows_by_epoch: dict[int, dict[str, float | int]] = {}
    for raw_row in raw_rows:
        row = coerce_history_row(raw_row)
        epoch = int(row["epoch"])
        if max_epoch is None or epoch <= max_epoch:
            rows_by_epoch[epoch] = row
    return [rows_by_epoch[epoch] for epoch in sorted(rows_by_epoch)]


def write_history_rows(csv_path: Path, history: list[dict[str, float | int]]) -> None:
    ensure_dir(csv_path.parent)
    tmp_path = csv_path.with_name(f".{csv_path.name}.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(history)
        handle.flush()
        os.fsync(handle.fileno())
    tmp_path.replace(csv_path)


def resume_progress_from_history(
    history: list[dict[str, float | int]],
) -> tuple[float, int, int, dict[str, float | int] | None]:
    best_val_iou = -1.0
    best_epoch = 0
    best_metrics = None
    epochs_without_improvement = 0
    for row in history:
        val_iou = float(row["val_iou"])
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_epoch = int(row["epoch"])
            best_metrics = row
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
    return best_val_iou, best_epoch, epochs_without_improvement, best_metrics


def move_optimizer_state_to_device(optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def maybe_freeze_encoder(model: torch.nn.Module, freeze: bool) -> None:
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return
    for parameter in encoder.parameters():
        parameter.requires_grad = not freeze


def main() -> None:
    parser = argparse.ArgumentParser(description="Train binary weed segmentation model.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Resume training from a checkpoint, usually outputs/runs/<run>/checkpoints/latest.pt.",
    )
    parser.add_argument(
        "--resume-run-dir",
        default=None,
        help="Shortcut for --resume-checkpoint <run_dir>/checkpoints/latest.pt.",
    )
    args = parser.parse_args()

    try:
        resume_checkpoint_path = resolve_resume_checkpoint(args.resume_checkpoint, args.resume_run_dir)
    except ValueError as exc:
        parser.error(str(exc))

    if args.config is None and resume_checkpoint_path is None:
        parser.error("--config is required unless --resume-checkpoint or --resume-run-dir is provided.")

    resume_payload: dict[str, Any] | None = None
    if resume_checkpoint_path is not None:
        resume_checkpoint_path = resume_checkpoint_path.resolve()
        if not resume_checkpoint_path.is_file():
            parser.error(f"Resume checkpoint not found: {resume_checkpoint_path}")
        resume_payload = torch.load(resume_checkpoint_path, map_location="cpu")
        if not isinstance(resume_payload, dict):
            parser.error(f"Resume checkpoint is not a training checkpoint: {resume_checkpoint_path}")

    if args.config is not None:
        config = load_config(args.config)
    else:
        assert resume_payload is not None
        checkpoint_config = resume_payload.get("config")
        if not isinstance(checkpoint_config, dict):
            parser.error(f"Resume checkpoint does not contain a config: {resume_checkpoint_path}")
        config = deepcopy(checkpoint_config)

    if args.seed is not None:
        config["seed"] = args.seed
    elif resume_payload is not None:
        checkpoint_config = resume_payload.get("config", {})
        if isinstance(checkpoint_config, dict) and checkpoint_config.get("seed") is not None:
            config["seed"] = checkpoint_config["seed"]
    config["seed"] = resolve_seed(config.get("seed"))
    set_seed(config["seed"])
    model_config = resolve_model_config(config.get("model"))
    config["model"] = model_config

    if resume_checkpoint_path is not None:
        try:
            run_dir = ensure_dir(run_dir_from_checkpoint(resume_checkpoint_path))
        except ValueError as exc:
            parser.error(str(exc))
    else:
        run_name = f"{config['experiment_name']}_seed{config['seed']}_{timestamp_utc()}"
        run_dir = ensure_dir(Path(config["paths"]["run_root"]) / run_name)
    checkpoint_dir = ensure_dir(run_dir / "checkpoints")
    log_dir = ensure_dir(run_dir / "logs")
    metrics_dir = ensure_dir(run_dir / "metrics")
    logger = setup_file_logger("Trainer", log_dir / "execution.log")

    dump_config(run_dir / "config.yaml", config)
    metadata_path = run_dir / "run_metadata.json"
    metadata = read_json(metadata_path) if metadata_path.is_file() else {}
    resume_count = int(metadata.get("resume_count", 0))
    metadata = {
        **metadata,
        "git_commit": get_git_commit(),
        "seed": config["seed"],
        "packages": package_versions(["torch", "numpy", "PIL"]),
        "run_kind": "finetuned",
    }
    if resume_checkpoint_path is not None:
        metadata.update(
            {
                "resume_count": resume_count + 1,
                "last_resume_checkpoint": str(resume_checkpoint_path),
                "last_resume_at": timestamp_utc(),
            }
        )
    write_json(metadata_path, metadata)
    if resume_checkpoint_path is not None:
        logger.info("Training resume initialized at %s", run_dir.resolve())
        logger.info("Resuming from checkpoint %s", resume_checkpoint_path)
    else:
        logger.info("Training run initialized at %s", run_dir.resolve())
    logger.info("Loaded config from %s", config.get("_config_path"))
    logger.info("Using seed=%s and device=%s", config["seed"], config.get("device", "auto"))
    logger.info(
        "Using model architecture=%s encoder=%s encoder_weights=%s",
        model_config["architecture"],
        model_config["encoder_name"],
        model_config["encoder_weights"],
    )

    normalization = prepare_normalization(config, run_dir)
    device = device_from_config(config.get("device"))
    logger.info("Resolved runtime device to %s", device)
    logger.info("Prepared normalization from source=%s", normalization.get("source", "unknown"))

    train_loader = build_dataloader(
        sample_manifest_path=config["paths"]["train_manifest"],
        patch_manifest_path=config["paths"]["train_patch_manifest"],
        modality=config["modality"],
        normalization=normalization,
        batch_size=int(config["training"]["batch_size"]),
        num_workers=int(config["training"]["num_workers"]),
        is_train=True,
        transform_config=config,
        seed=config["seed"],
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
        seed=config["seed"],
    )

    model = build_model(int(config["in_channels"]), model_config).to(device)
    criterion = BCEDiceLoss(
        bce_weight=float(config["loss"]["bce_weight"]),
        dice_weight=float(config["loss"]["dice_weight"]),
    )
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=int(config["training"]["scheduler"].get("t_max", config["training"]["epochs"])),
        eta_min=float(config["training"]["scheduler"].get("eta_min", 0.0)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    logger.info(
        "Built dataloaders with batch_size=%s, num_workers=%s",
        config["training"]["batch_size"],
        config["training"]["num_workers"],
    )

    freeze_epochs = int(config["training"].get("freeze_encoder_epochs", 0))
    csv_path = log_dir / "history.csv"
    history_json_path = metrics_dir / "history.json"
    best_val_iou = -1.0
    best_epoch = 0
    best_metrics: dict[str, Any] | None = None
    epochs_without_improvement = 0
    resume_epoch = 0
    start_epoch = 1
    last_completed_epoch = 0
    if resume_payload is not None:
        model.load_state_dict(resume_payload["model_state_dict"])
        if resume_payload.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
            move_optimizer_state_to_device(optimizer, device)
        if resume_payload.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
        if resume_payload.get("scaler_state_dict") is not None:
            scaler.load_state_dict(resume_payload["scaler_state_dict"])

        resume_epoch = int(resume_payload.get("epoch", 0))
        start_epoch = resume_epoch + 1
        last_completed_epoch = resume_epoch
        logger.info("Loaded checkpoint state from epoch %s; next epoch is %s", resume_epoch, start_epoch)

    history = read_history_rows(csv_path, history_json_path, max_epoch=resume_epoch if resume_payload is not None else None)
    if resume_payload is not None:
        if history:
            best_val_iou, best_epoch, epochs_without_improvement, best_metrics = resume_progress_from_history(history)
        else:
            best_val_iou = float(resume_payload.get("best_val_iou", -1.0))
            best_epoch = int(resume_payload.get("best_epoch", 0))
            checkpoint_best_metrics = resume_payload.get("best_metrics")
            best_metrics = checkpoint_best_metrics if isinstance(checkpoint_best_metrics, dict) else None
            epochs_without_improvement = int(resume_payload.get("epochs_without_improvement", 0))
        logger.info(
            "Restored early-stopping progress: best_val_iou=%.6f best_epoch=%s epochs_without_improvement=%s",
            best_val_iou,
            best_epoch,
            epochs_without_improvement,
        )

    threshold = float(config["evaluation"]["threshold"])
    early_stopping_patience = int(config["training"]["early_stopping_patience"])
    total_epochs = int(config["training"]["epochs"])

    write_history_rows(csv_path, history)
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=HISTORY_FIELDNAMES,
        )

        if start_epoch > total_epochs:
            logger.info(
                "Checkpoint epoch %s already reached configured training.epochs=%s; skipping training loop",
                resume_epoch,
                total_epochs,
            )

        for epoch in range(start_epoch, total_epochs + 1):
            logger.info("Starting epoch %s", epoch)
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

            scheduler.step()
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
            os.fsync(handle.fileno())
            history.append(row)
            last_completed_epoch = epoch
            logger.info(
                "Completed epoch %s: train_loss=%.6f train_iou=%.6f val_loss=%.6f val_iou=%.6f lr=%.8f",
                epoch,
                train_metrics["loss"],
                train_metrics["iou"],
                val_metrics["loss"],
                val_metrics["iou"],
                current_lr,
            )

            if val_metrics["iou"] > best_val_iou:
                best_val_iou = val_metrics["iou"]
                best_epoch = epoch
                best_metrics = row
                epochs_without_improvement = 0
                save_checkpoint(
                    checkpoint_dir / "best.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    best_val_iou,
                    best_epoch,
                    best_metrics,
                    epochs_without_improvement,
                    config,
                )
                logger.info("New best checkpoint saved at epoch %s with val_iou=%.6f", best_epoch, best_val_iou)
            else:
                epochs_without_improvement += 1
                logger.info(
                    "No validation IoU improvement for %s epoch(s)",
                    epochs_without_improvement,
                )

            save_checkpoint(
                checkpoint_dir / "latest.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_val_iou,
                best_epoch,
                best_metrics,
                epochs_without_improvement,
                config,
            )
            logger.info("Updated latest checkpoint at epoch %s", epoch)

            if epochs_without_improvement >= early_stopping_patience:
                logger.info(
                    "Early stopping triggered after %s epoch(s) without improvement",
                    epochs_without_improvement,
                )
                break

    best_checkpoint_path = checkpoint_dir / "best.pt"
    latest_checkpoint_path = checkpoint_dir / "latest.pt"
    if not best_checkpoint_path.is_file() and latest_checkpoint_path.is_file():
        logger.info("Best checkpoint is missing; using the latest checkpoint as the best checkpoint for evaluation")
        save_checkpoint(
            best_checkpoint_path,
            model,
            optimizer,
            scheduler,
            scaler,
            last_completed_epoch,
            best_val_iou,
            best_epoch,
            best_metrics,
            epochs_without_improvement,
            config,
        )
    if not best_checkpoint_path.is_file():
        raise RuntimeError(f"Training did not create a best checkpoint: {best_checkpoint_path}")

    test_summary = evaluate_test_split(
        checkpoint_path=best_checkpoint_path,
        config=config,
        normalization=normalization,
        device=device,
        threshold=threshold,
    )

    write_json(metrics_dir / "history.json", history)
    write_json(metrics_dir / "test_summary.json", test_summary)
    logger.info("Wrote training history to %s", metrics_dir / "history.json")
    logger.info("Wrote test summary to %s", metrics_dir / "test_summary.json")
    logger.info("Training completed")
    print(str(run_dir.resolve()))


if __name__ == "__main__":
    main()
