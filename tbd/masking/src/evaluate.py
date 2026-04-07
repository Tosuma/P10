from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.datasets import build_dataloader, load_image_for_modality, load_mask
from src.losses import BCEDiceLoss
from src.metrics import ConfidenceTotals, ConfusionTotals
from src.model import build_model, resolve_model_config
from src.utils import (
    device_from_config,
    ensure_dir,
    read_csv_rows,
    read_json,
    resolve_seed,
    setup_file_logger,
    write_csv_rows,
    write_json,
)
from src.visualize import save_prediction_panel


def load_checkpoint(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    config["model"] = resolve_model_config(config.get("model"))
    model = build_model(int(config["in_channels"]), config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config, checkpoint


def evaluate_loader(model, loader, device: torch.device, threshold: float):
    totals = ConfusionTotals()
    confidence_totals = ConfidenceTotals()
    per_patch_rows = []
    probabilities_by_sample = defaultdict(list)
    criterion = BCEDiceLoss()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].unsqueeze(1).to(device)
            logits = model(images)
            loss = criterion(logits, masks)
            probs = torch.sigmoid(logits).cpu()
            preds = (probs >= threshold).float()
            total_loss += loss.item()
            total_batches += 1
            totals.update(preds, masks.cpu())
            confidence_totals.update(probs, threshold)

            for idx, metadata in enumerate(batch["metadata"]):
                patch_totals = ConfusionTotals()
                patch_confidence_totals = ConfidenceTotals()
                patch_totals.update(preds[idx : idx + 1], masks[idx : idx + 1].cpu())
                patch_confidence_totals.update(probs[idx : idx + 1], threshold)
                metrics = patch_totals.compute()
                confidence_metrics = patch_confidence_totals.compute()
                per_patch_rows.append(
                    {
                        "sample_id": metadata["sample_id"],
                        "patch_id": metadata["patch_id"],
                        "x": metadata["coords"][0],
                        "y": metadata["coords"][1],
                        "width": metadata["coords"][2],
                        "height": metadata["coords"][3],
                        "iou": metrics["iou"],
                        "dice": metrics["dice"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "confidence_mean": confidence_metrics["confidence_mean"],
                        "positive_confidence_mean": confidence_metrics["positive_confidence_mean"],
                        "negative_confidence_mean": confidence_metrics["negative_confidence_mean"],
                    }
                )
                probabilities_by_sample[metadata["sample_id"]].append(
                    {"coords": metadata["coords"], "probability": probs[idx, 0].numpy()}
                )
    return {
        "overall": {
            **totals.compute(),
            **confidence_totals.compute(),
            "loss": total_loss / max(total_batches, 1),
        },
        "per_patch_rows": per_patch_rows,
        "reconstruction_payload": probabilities_by_sample,
    }


def reconstruct_image_metrics(sample_rows: list[dict[str, str]], probabilities_by_sample: dict, threshold: float):
    sample_lookup = {row["sample_id"]: row for row in sample_rows}
    per_image_rows = []
    visuals = []
    for sample_id, patches in probabilities_by_sample.items():
        sample = sample_lookup[sample_id]
        gt_mask = load_mask(sample["mask_path"])
        height, width = gt_mask.shape
        accumulator = np.zeros((height, width), dtype=np.float32)
        counts = np.zeros((height, width), dtype=np.float32)
        for patch in patches:
            left, top, patch_width, patch_height = patch["coords"]
            accumulator[top : top + patch_height, left : left + patch_width] += patch["probability"]
            counts[top : top + patch_height, left : left + patch_width] += 1.0
        probability_map = accumulator / np.clip(counts, 1e-6, None)
        pred_mask = (probability_map >= threshold).astype(np.float32)
        totals = ConfusionTotals()
        confidence_totals = ConfidenceTotals()
        totals.update(torch.from_numpy(pred_mask[None, None, ...]), torch.from_numpy(gt_mask[None, None, ...]))
        confidence_totals.update(torch.from_numpy(probability_map[None, None, ...]), threshold)
        metrics = totals.compute()
        per_image_rows.append({"sample_id": sample_id, **metrics, **confidence_totals.compute()})
        visuals.append(
            {
                "sample_id": sample_id,
                "gt_mask": gt_mask,
                "probability_map": probability_map,
                "pred_mask": pred_mask,
            }
        )
    return per_image_rows, visuals


def load_preview_image(sample: dict[str, str], modality: str) -> np.ndarray:
    return load_image_for_modality(sample, modality)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained segmentation checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    device = device_from_config()
    model, config, _ = load_checkpoint(args.checkpoint, device)
    run_dir = Path(args.checkpoint).resolve().parents[1]
    output_dir = ensure_dir(args.output_dir or (run_dir / "evaluation" / args.split))
    logger = setup_file_logger("Evaluator", output_dir / "execution.log")
    logger.info("Evaluation started for checkpoint %s", Path(args.checkpoint).resolve())
    logger.info("Writing evaluation artifacts to %s", output_dir.resolve())
    logger.info("Using split=%s and device=%s", args.split, device)

    normalization = read_json(Path(config["paths"]["normalization_stats"]))
    resolved_seed = resolve_seed(config.get("seed"))
    loader = build_dataloader(
        sample_manifest_path=config["paths"][f"{args.split}_manifest"],
        patch_manifest_path=config["paths"][f"{args.split}_patch_manifest"],
        modality=config["modality"],
        normalization=normalization,
        batch_size=int(config["training"]["batch_size"]),
        num_workers=int(config["training"]["num_workers"]),
        is_train=False,
        transform_config=config,
        seed=resolved_seed,
    )

    threshold = float(config["evaluation"]["threshold"])
    logger.info("Loaded normalization from %s", config["paths"]["normalization_stats"])
    logger.info("Running inference with threshold=%.4f", threshold)
    results = evaluate_loader(model, loader, device, threshold=threshold)
    sample_rows = read_csv_rows(config["paths"][f"{args.split}_manifest"])
    per_image_rows, visuals = reconstruct_image_metrics(sample_rows, results["reconstruction_payload"], threshold)
    image_summary = {
        "mean_iou": float(np.mean([row["iou"] for row in per_image_rows])) if per_image_rows else 0.0,
        "mean_dice": float(np.mean([row["dice"] for row in per_image_rows])) if per_image_rows else 0.0,
        "mean_precision": float(np.mean([row["precision"] for row in per_image_rows])) if per_image_rows else 0.0,
        "mean_recall": float(np.mean([row["recall"] for row in per_image_rows])) if per_image_rows else 0.0,
        "mean_accuracy": float(np.mean([row["accuracy"] for row in per_image_rows])) if per_image_rows else 0.0,
        "mean_specificity": float(np.mean([row["specificity"] for row in per_image_rows])) if per_image_rows else 0.0,
        "mean_confidence": float(np.mean([row["confidence_mean"] for row in per_image_rows])) if per_image_rows else 0.0,
        "mean_positive_confidence": float(np.mean([row["positive_confidence_mean"] for row in per_image_rows])) if per_image_rows else 0.0,
        "mean_negative_confidence": float(np.mean([row["negative_confidence_mean"] for row in per_image_rows])) if per_image_rows else 0.0,
    }

    write_json(output_dir / "overall_metrics.json", {"patch_level": results["overall"], "image_level": image_summary})
    write_csv_rows(output_dir / "per_patch_metrics.csv", results["per_patch_rows"], list(results["per_patch_rows"][0].keys()) if results["per_patch_rows"] else ["sample_id"])
    write_csv_rows(output_dir / "per_image_metrics.csv", per_image_rows, list(per_image_rows[0].keys()) if per_image_rows else ["sample_id"])
    logger.info("Saved aggregate metrics to %s", output_dir / "overall_metrics.json")
    logger.info("Saved %s patch rows and %s image rows", len(results["per_patch_rows"]), len(per_image_rows))

    preview_dir = ensure_dir(output_dir / "visuals")
    sample_lookup = {row["sample_id"]: row for row in sample_rows}
    for visual in visuals[: int(config["evaluation"].get("num_visualizations", 8))]:
        sample = sample_lookup[visual["sample_id"]]
        image = load_preview_image(sample, config["modality"])
        save_prediction_panel(
            preview_dir / f"{visual['sample_id']}.png",
            image=image,
            gt_mask=visual["gt_mask"],
            probability_map=visual["probability_map"],
            pred_mask=visual["pred_mask"],
        )
    logger.info("Saved %s visualization panel(s) to %s", min(len(visuals), int(config["evaluation"].get("num_visualizations", 8))), preview_dir)
    logger.info("Evaluation completed")


if __name__ == "__main__":
    main()
