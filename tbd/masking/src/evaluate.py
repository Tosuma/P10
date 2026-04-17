from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.datasets import build_dataloader, load_image_for_modality, load_mask
from src.losses import BCEDiceLoss
from src.metrics import ConfidenceTotals, ConfusionTotals
from src.model import build_model, resolve_model_config
from src.targets import resolve_target_config, target_views_from_hard_mask
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
from src.visualize import save_binary_mask, save_prediction_panel


def _compute_patch_metrics(preds: torch.Tensor, masks: torch.Tensor, probs: torch.Tensor, threshold: float) -> dict[str, float]:
    patch_totals = ConfusionTotals()
    patch_confidence_totals = ConfidenceTotals()
    patch_totals.update(preds, masks)
    patch_confidence_totals.update(probs, threshold)
    return {
        **patch_totals.compute(),
        **patch_confidence_totals.compute(),
    }


def load_checkpoint(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    config["model"] = resolve_model_config(config.get("model"))
    model = build_model(int(config["in_channels"]), config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config, checkpoint


def evaluate_loader(model, loader, device: torch.device, threshold: float, config: dict[str, Any]):
    target_config = resolve_target_config(config)
    primary_view = "relaxed" if target_config["mode"] == "fuzzy_halo" else "original"
    totals_by_view = {view: ConfusionTotals() for view in ("original", "relaxed")}
    confidence_totals = ConfidenceTotals()
    per_patch_rows_by_view = {view: [] for view in ("original", "relaxed")}
    probabilities_by_sample = defaultdict(list)
    criterion = BCEDiceLoss()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            target_masks = batch["mask"].unsqueeze(1).to(device)
            hard_masks = batch["hard_mask"].unsqueeze(1).to(device)
            logits = model(images)
            loss = criterion(logits, target_masks)
            probs = torch.sigmoid(logits).cpu()
            preds = (probs >= threshold).float()
            total_loss += loss.item()
            total_batches += 1
            totals_by_view["original"].update(preds, hard_masks.cpu())
            totals_by_view["relaxed"].update(preds, target_masks.cpu())
            confidence_totals.update(probs, threshold)

            for idx, metadata in enumerate(batch["metadata"]):
                original_metrics = _compute_patch_metrics(
                    preds[idx : idx + 1],
                    hard_masks[idx : idx + 1].cpu(),
                    probs[idx : idx + 1],
                    threshold,
                )
                relaxed_metrics = _compute_patch_metrics(
                    preds[idx : idx + 1],
                    target_masks[idx : idx + 1].cpu(),
                    probs[idx : idx + 1],
                    threshold,
                )
                base_row = {
                    "sample_id": metadata["sample_id"],
                    "patch_id": metadata["patch_id"],
                    "x": metadata["coords"][0],
                    "y": metadata["coords"][1],
                    "width": metadata["coords"][2],
                    "height": metadata["coords"][3],
                }
                per_patch_rows_by_view["original"].append(
                    {
                        **base_row,
                        **original_metrics,
                    }
                )
                per_patch_rows_by_view["relaxed"].append(
                    {
                        **base_row,
                        **relaxed_metrics,
                    }
                )
                probabilities_by_sample[metadata["sample_id"]].append(
                    {"coords": metadata["coords"], "probability": probs[idx, 0].numpy()}
                )
    return {
        "target_mode": target_config["mode"],
        "primary_view": primary_view,
        "views": {
            view: {
                "overall": {
                    **totals_by_view[view].compute(),
                    **confidence_totals.compute(),
                    "loss": total_loss / max(total_batches, 1),
                },
                "per_patch_rows": per_patch_rows_by_view[view],
            }
            for view in ("original", "relaxed")
        },
        "reconstruction_payload": probabilities_by_sample,
    }


def reconstruct_image_metrics(sample_rows: list[dict[str, str]], probabilities_by_sample: dict, threshold: float, config: dict[str, Any]):
    sample_lookup = {row["sample_id"]: row for row in sample_rows}
    target_config = resolve_target_config(config)
    primary_view = "relaxed" if target_config["mode"] == "fuzzy_halo" else "original"
    per_image_rows_by_view = {"original": [], "relaxed": []}
    visuals = []
    for sample_id, patches in probabilities_by_sample.items():
        sample = sample_lookup[sample_id]
        hard_gt_mask = load_mask(sample["mask_path"])
        target_views = target_views_from_hard_mask(hard_gt_mask, config)
        height, width = hard_gt_mask.shape
        accumulator = np.zeros((height, width), dtype=np.float32)
        counts = np.zeros((height, width), dtype=np.float32)
        for patch in patches:
            left, top, patch_width, patch_height = patch["coords"]
            accumulator[top : top + patch_height, left : left + patch_width] += patch["probability"]
            counts[top : top + patch_height, left : left + patch_width] += 1.0
        probability_map = accumulator / np.clip(counts, 1e-6, None)
        pred_mask = (probability_map >= threshold).astype(np.float32)
        confidence_totals = ConfidenceTotals()
        confidence_totals.update(torch.from_numpy(probability_map[None, None, ...]), threshold)
        confidence_metrics = confidence_totals.compute()
        for view in ("original", "relaxed"):
            totals = ConfusionTotals()
            gt_mask = target_views[view]
            totals.update(torch.from_numpy(pred_mask[None, None, ...]), torch.from_numpy(gt_mask[None, None, ...]))
            per_image_rows_by_view[view].append(
                {"sample_id": sample_id, **totals.compute(), **confidence_metrics}
            )
        visuals.append(
            {
                "sample_id": sample_id,
                "gt_mask": hard_gt_mask,
                "probability_map": probability_map,
                "pred_mask": pred_mask,
            }
        )
    return {
        "primary_view": primary_view,
        "views": per_image_rows_by_view,
        "visuals": visuals,
    }


def summarize_metric_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    if not rows:
        return summary

    excluded_keys = {"sample_id", "patch_id", "x", "y", "width", "height"}
    metric_keys = [
        key
        for key, value in rows[0].items()
        if key not in excluded_keys and isinstance(value, int | float)
    ]
    for key in metric_keys:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), int | float)]
        if not values:
            continue
        summary[f"{key}_avg"] = float(np.mean(values))
        summary[f"{key}_median"] = float(np.median(values))
    return summary


def mean_aliases(summary: dict[str, float]) -> dict[str, float]:
    aliases = {}
    legacy_names = {
        "confidence_mean": "confidence",
        "positive_confidence_mean": "positive_confidence",
        "negative_confidence_mean": "negative_confidence",
    }
    for key, value in summary.items():
        if not key.endswith("_avg"):
            continue
        metric_name = key.removesuffix("_avg")
        aliases[f"mean_{legacy_names.get(metric_name, metric_name)}"] = value
    return aliases


def load_preview_image(sample: dict[str, str], modality: str) -> np.ndarray:
    return load_image_for_modality(sample, modality)


def run_split_evaluation(
    model,
    config: dict[str, Any],
    split: str,
    output_dir: Path,
    device: torch.device,
    logger,
) -> dict[str, Any]:
    normalization = read_json(Path(config["paths"]["normalization_stats"]))
    resolved_seed = resolve_seed(config.get("seed"))
    loader = build_dataloader(
        sample_manifest_path=config["paths"][f"{split}_manifest"],
        patch_manifest_path=config["paths"][f"{split}_patch_manifest"],
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
    results = evaluate_loader(model, loader, device, threshold=threshold, config=config)
    sample_rows = read_csv_rows(config["paths"][f"{split}_manifest"])
    image_results = reconstruct_image_metrics(sample_rows, results["reconstruction_payload"], threshold, config=config)
    primary_view = results["primary_view"]
    secondary_view = "original" if primary_view == "relaxed" else None
    primary_patch_rows = results["views"][primary_view]["per_patch_rows"]
    primary_image_rows = image_results["views"][primary_view]
    patch_summary = summarize_metric_rows(primary_patch_rows)
    image_summary = summarize_metric_rows(primary_image_rows)

    overall_payload = {
        "target_mode": results["target_mode"],
        "patch_level": results["views"][primary_view]["overall"],
        "patch_summary": patch_summary,
        "image_level": {**mean_aliases(image_summary), **image_summary},
    }
    if secondary_view is not None:
        secondary_patch_rows = results["views"][secondary_view]["per_patch_rows"]
        secondary_image_rows = image_results["views"][secondary_view]
        secondary_patch_summary = summarize_metric_rows(secondary_patch_rows)
        secondary_image_summary = summarize_metric_rows(secondary_image_rows)
        overall_payload[f"{secondary_view}_patch_level"] = results["views"][secondary_view]["overall"]
        overall_payload[f"{secondary_view}_patch_summary"] = secondary_patch_summary
        overall_payload[f"{secondary_view}_image_level"] = {
            **mean_aliases(secondary_image_summary),
            **secondary_image_summary,
        }

    write_json(output_dir / "overall_metrics.json", overall_payload)
    write_csv_rows(output_dir / "per_patch_metrics.csv", primary_patch_rows, list(primary_patch_rows[0].keys()) if primary_patch_rows else ["sample_id"])
    write_csv_rows(output_dir / "per_image_metrics.csv", primary_image_rows, list(primary_image_rows[0].keys()) if primary_image_rows else ["sample_id"])
    if secondary_view is not None:
        secondary_patch_rows = results["views"][secondary_view]["per_patch_rows"]
        secondary_image_rows = image_results["views"][secondary_view]
        write_csv_rows(
            output_dir / f"{secondary_view}_per_patch_metrics.csv",
            secondary_patch_rows,
            list(secondary_patch_rows[0].keys()) if secondary_patch_rows else ["sample_id"],
        )
        write_csv_rows(
            output_dir / f"{secondary_view}_per_image_metrics.csv",
            secondary_image_rows,
            list(secondary_image_rows[0].keys()) if secondary_image_rows else ["sample_id"],
        )
    logger.info("Saved aggregate metrics to %s", output_dir / "overall_metrics.json")
    logger.info("Saved %s patch rows and %s image rows", len(primary_patch_rows), len(primary_image_rows))

    mask_dir = ensure_dir(output_dir / "masks")
    for visual in image_results["visuals"]:
        save_binary_mask(mask_dir / f"{visual['sample_id']}.png", visual["pred_mask"])
    logger.info("Saved %s predicted mask(s) to %s", len(image_results["visuals"]), mask_dir)

    preview_dir = ensure_dir(output_dir / "visuals")
    sample_lookup = {row["sample_id"]: row for row in sample_rows}
    for visual in image_results["visuals"][: int(config["evaluation"].get("num_visualizations", 8))]:
        sample = sample_lookup[visual["sample_id"]]
        image = load_preview_image(sample, config["modality"])
        save_prediction_panel(
            preview_dir / f"{visual['sample_id']}.png",
            image=image,
            gt_mask=visual["gt_mask"],
            probability_map=visual["probability_map"],
            pred_mask=visual["pred_mask"],
        )
    logger.info("Saved %s visualization panel(s) to %s", min(len(image_results["visuals"]), int(config["evaluation"].get("num_visualizations", 8))), preview_dir)
    return overall_payload


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

    run_split_evaluation(model, config, args.split, output_dir, device, logger)
    logger.info("Evaluation completed")


if __name__ == "__main__":
    main()
