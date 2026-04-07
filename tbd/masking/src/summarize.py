from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from src.config import load_config
from src.utils import read_json, setup_file_logger, write_json

AGGREGATE_METRICS = [
    "val_loss",
    "val_iou",
    "val_dice",
    "val_precision",
    "val_recall",
    "test_patch_loss",
    "test_patch_iou",
    "test_patch_dice",
    "test_patch_precision",
    "test_patch_recall",
    "test_patch_accuracy",
    "test_patch_specificity",
    "test_patch_confidence",
    "test_patch_positive_confidence",
    "test_patch_negative_confidence",
    "test_image_iou",
    "test_image_dice",
    "test_image_precision",
    "test_image_recall",
    "test_image_accuracy",
    "test_image_specificity",
    "test_image_confidence",
    "test_image_positive_confidence",
    "test_image_negative_confidence",
]


def summarize_history(history: list[dict]) -> dict:
    if not history:
        return {
            "best_epoch": 0,
            "val_loss": 0.0,
            "val_iou": 0.0,
            "val_dice": 0.0,
            "val_precision": 0.0,
            "val_recall": 0.0,
        }

    best_row = max(history, key=lambda row: float(row["val_iou"]))
    return {
        "best_epoch": int(best_row["epoch"]),
        "val_loss": float(best_row["val_loss"]),
        "val_iou": float(best_row["val_iou"]),
        "val_dice": float(best_row["val_dice"]),
        "val_precision": float(best_row["val_precision"]),
        "val_recall": float(best_row["val_recall"]),
    }


def build_aggregate_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, str], list[dict]] = {}
    for row in rows:
        key = (row["experiment_name"], row["modality"], row["architecture"], row["encoder_name"])
        grouped.setdefault(key, []).append(row)

    aggregate_rows = []
    for (experiment_name, modality, architecture, encoder_name), group_rows in sorted(grouped.items()):
        aggregate_row = {
            "experiment_name": experiment_name,
            "modality": modality,
            "architecture": architecture,
            "encoder_name": encoder_name,
            "runs": len(group_rows),
        }
        for metric in AGGREGATE_METRICS:
            values = [float(row[metric]) for row in group_rows]
            aggregate_row[f"{metric}_mean"] = statistics.fmean(values)
            aggregate_row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        aggregate_rows.append(aggregate_row)
    return aggregate_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize modality experiment metrics into one comparison table.")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_path = Path(args.output)
    logger = setup_file_logger("Summarizer", output_path.with_suffix(".log"))
    logger.info("Summary generation started")
    logger.info("Collecting metrics from %s run(s)", len(args.runs))

    rows = []
    for run_dir in args.runs:
        run_path = Path(run_dir)
        logger.info("Reading run artifacts from %s", run_path.resolve())
        config = load_config(run_path / "config.yaml")
        metadata = read_json(run_path / "run_metadata.json")
        eval_metrics = read_json(run_path / "evaluation" / "test" / "overall_metrics.json")
        history = read_json(run_path / "metrics" / "history.json")
        best_summary = summarize_history(history)
        model_config = config.get("model", {})
        rows.append(
            {
                "experiment_name": config["experiment_name"],
                "modality": config["modality"],
                "architecture": model_config.get("architecture", "Unet"),
                "encoder_name": model_config.get("encoder_name", "resnet34"),
                "seed": metadata["seed"],
                "best_epoch": best_summary["best_epoch"],
                "val_loss": best_summary["val_loss"],
                "val_iou": best_summary["val_iou"],
                "val_dice": best_summary["val_dice"],
                "val_precision": best_summary["val_precision"],
                "val_recall": best_summary["val_recall"],
                "test_patch_loss": eval_metrics["patch_level"]["loss"],
                "test_patch_iou": eval_metrics["patch_level"]["iou"],
                "test_patch_dice": eval_metrics["patch_level"]["dice"],
                "test_patch_precision": eval_metrics["patch_level"]["precision"],
                "test_patch_recall": eval_metrics["patch_level"]["recall"],
                "test_patch_accuracy": eval_metrics["patch_level"]["accuracy"],
                "test_patch_specificity": eval_metrics["patch_level"]["specificity"],
                "test_patch_confidence": eval_metrics["patch_level"].get("confidence_mean", 0.0),
                "test_patch_positive_confidence": eval_metrics["patch_level"].get("positive_confidence_mean", 0.0),
                "test_patch_negative_confidence": eval_metrics["patch_level"].get("negative_confidence_mean", 0.0),
                "test_image_iou": eval_metrics["image_level"]["mean_iou"],
                "test_image_dice": eval_metrics["image_level"]["mean_dice"],
                "test_image_precision": eval_metrics["image_level"]["mean_precision"],
                "test_image_recall": eval_metrics["image_level"]["mean_recall"],
                "test_image_accuracy": eval_metrics["image_level"].get("mean_accuracy", 0.0),
                "test_image_specificity": eval_metrics["image_level"].get("mean_specificity", 0.0),
                "test_image_confidence": eval_metrics["image_level"].get("mean_confidence", 0.0),
                "test_image_positive_confidence": eval_metrics["image_level"].get("mean_positive_confidence", 0.0),
                "test_image_negative_confidence": eval_metrics["image_level"].get("mean_negative_confidence", 0.0),
            }
        )
    aggregate_rows = build_aggregate_rows(rows)
    payload = {
        "runs": rows,
        "aggregates": aggregate_rows,
    }
    write_json(output_path, payload)
    logger.info("Wrote %s run summary item(s) to %s", len(rows), output_path.resolve())
    logger.info("Wrote %s aggregate summary item(s)", len(aggregate_rows))
    logger.info("Summary generation completed")


if __name__ == "__main__":
    main()
