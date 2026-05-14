#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

MASKING_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MASKING_ROOT))

from src.config import load_config
from src.summarize import build_aggregate_rows, summarize_history
from src.utils import read_json, write_json


def read_optional_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return read_json(path)


def mean_metric(metrics: dict[str, Any], name: str, default: float = 0.0) -> float:
    return float(metrics.get(f"mean_{name}", metrics.get(f"{name}_avg", default)))


def completed_run_dirs(runs_root: Path) -> list[Path]:
    if not runs_root.is_dir():
        raise SystemExit(f"Runs root was not found: {runs_root}")
    run_dirs = []
    for metrics_path in runs_root.glob("*/evaluation/test/overall_metrics.json"):
        run_dir = metrics_path.parents[2]
        if (run_dir / "config.yaml").is_file():
            run_dirs.append(run_dir)
    return sorted(run_dirs)


def run_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["run_kind"],
        row["experiment_name"],
        row["modality"],
        row["architecture"],
        row["encoder_name"],
        row["target_mode"],
        row["halo_radius_px"],
        row["halo_min_value"],
        row["seed"],
    )


def build_run_row(run_path: Path) -> dict[str, Any]:
    config = load_config(run_path / "config.yaml")
    metadata = read_optional_json(run_path / "run_metadata.json", {})
    eval_metrics = read_json(run_path / "evaluation" / "test" / "overall_metrics.json")
    history = read_optional_json(run_path / "metrics" / "history.json", [])
    best_summary = summarize_history(history)
    model_config = config.get("model", {})
    target_config = config.get("target", {})
    original_patch_level = eval_metrics.get("original_patch_level", eval_metrics["patch_level"])
    original_image_level = eval_metrics.get("original_image_level", eval_metrics["image_level"])
    fuzzy_patch_level = eval_metrics.get("fuzzy_patch_level", eval_metrics["patch_level"])
    fuzzy_image_level = eval_metrics.get("fuzzy_image_level", eval_metrics["image_level"])

    return {
        "run_dir": str(run_path),
        "run_kind": metadata.get("run_kind", "finetuned"),
        "experiment_name": config["experiment_name"],
        "modality": config["modality"],
        "architecture": model_config.get("architecture", "Unet"),
        "encoder_name": model_config.get("encoder_name", "resnet34"),
        "target_mode": target_config.get("mode", "binary"),
        "halo_radius_px": int(target_config.get("halo_radius_px", 0)),
        "halo_min_value": float(target_config.get("halo_min_value", 0.0)),
        "seed": int(metadata.get("seed", config.get("seed", 0))),
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
        "test_image_iou": mean_metric(eval_metrics["image_level"], "iou"),
        "test_image_dice": mean_metric(eval_metrics["image_level"], "dice"),
        "test_image_precision": mean_metric(eval_metrics["image_level"], "precision"),
        "test_image_recall": mean_metric(eval_metrics["image_level"], "recall"),
        "test_image_accuracy": mean_metric(eval_metrics["image_level"], "accuracy"),
        "test_image_specificity": mean_metric(eval_metrics["image_level"], "specificity"),
        "test_image_confidence": mean_metric(eval_metrics["image_level"], "confidence"),
        "test_image_positive_confidence": mean_metric(eval_metrics["image_level"], "positive_confidence"),
        "test_image_negative_confidence": mean_metric(eval_metrics["image_level"], "negative_confidence"),
        "original_test_patch_iou": original_patch_level["iou"],
        "original_test_patch_dice": original_patch_level["dice"],
        "original_test_patch_precision": original_patch_level["precision"],
        "original_test_patch_recall": original_patch_level["recall"],
        "original_test_patch_accuracy": original_patch_level["accuracy"],
        "original_test_patch_specificity": original_patch_level["specificity"],
        "original_test_patch_confidence": original_patch_level.get("confidence_mean", 0.0),
        "original_test_patch_positive_confidence": original_patch_level.get("positive_confidence_mean", 0.0),
        "original_test_patch_negative_confidence": original_patch_level.get("negative_confidence_mean", 0.0),
        "original_test_image_iou": mean_metric(original_image_level, "iou"),
        "original_test_image_dice": mean_metric(original_image_level, "dice"),
        "original_test_image_precision": mean_metric(original_image_level, "precision"),
        "original_test_image_recall": mean_metric(original_image_level, "recall"),
        "original_test_image_accuracy": mean_metric(original_image_level, "accuracy"),
        "original_test_image_specificity": mean_metric(original_image_level, "specificity"),
        "original_test_image_confidence": mean_metric(original_image_level, "confidence"),
        "original_test_image_positive_confidence": mean_metric(original_image_level, "positive_confidence"),
        "original_test_image_negative_confidence": mean_metric(original_image_level, "negative_confidence"),
        "fuzzy_test_patch_iou": fuzzy_patch_level["iou"],
        "fuzzy_test_patch_dice": fuzzy_patch_level["dice"],
        "fuzzy_test_patch_precision": fuzzy_patch_level["precision"],
        "fuzzy_test_patch_recall": fuzzy_patch_level["recall"],
        "fuzzy_test_patch_accuracy": fuzzy_patch_level["accuracy"],
        "fuzzy_test_patch_specificity": fuzzy_patch_level["specificity"],
        "fuzzy_test_patch_confidence": fuzzy_patch_level.get("confidence_mean", 0.0),
        "fuzzy_test_patch_positive_confidence": fuzzy_patch_level.get("positive_confidence_mean", 0.0),
        "fuzzy_test_patch_negative_confidence": fuzzy_patch_level.get("negative_confidence_mean", 0.0),
        "fuzzy_test_image_iou": mean_metric(fuzzy_image_level, "iou"),
        "fuzzy_test_image_dice": mean_metric(fuzzy_image_level, "dice"),
        "fuzzy_test_image_precision": mean_metric(fuzzy_image_level, "precision"),
        "fuzzy_test_image_recall": mean_metric(fuzzy_image_level, "recall"),
        "fuzzy_test_image_accuracy": mean_metric(fuzzy_image_level, "accuracy"),
        "fuzzy_test_image_specificity": mean_metric(fuzzy_image_level, "specificity"),
        "fuzzy_test_image_confidence": mean_metric(fuzzy_image_level, "confidence"),
        "fuzzy_test_image_positive_confidence": mean_metric(fuzzy_image_level, "positive_confidence"),
        "fuzzy_test_image_negative_confidence": mean_metric(fuzzy_image_level, "negative_confidence"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize all completed masking runs under a runs root.")
    parser.add_argument("--runs-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs-list-output", type=Path, default=None)
    args = parser.parse_args()

    rows_by_identity: dict[tuple[Any, ...], dict[str, Any]] = {}
    mtime_by_identity: dict[tuple[Any, ...], float] = {}
    for run_dir in completed_run_dirs(args.runs_root):
        row = build_run_row(run_dir)
        identity = run_identity(row)
        metrics_mtime = (run_dir / "evaluation" / "test" / "overall_metrics.json").stat().st_mtime
        if identity not in rows_by_identity or metrics_mtime >= mtime_by_identity[identity]:
            rows_by_identity[identity] = row
            mtime_by_identity[identity] = metrics_mtime

    rows = [rows_by_identity[key] for key in sorted(rows_by_identity)]
    if not rows:
        raise SystemExit(f"No completed runs with config.yaml and evaluation/test/overall_metrics.json found under {args.runs_root}")

    write_json(args.output, {"runs": rows, "aggregates": build_aggregate_rows(rows)})
    runs_list_output = args.runs_list_output or args.output.with_name(f"{args.output.stem}_runs.txt")
    runs_list_output.parent.mkdir(parents=True, exist_ok=True)
    runs_list_output.write_text("\n".join(row["run_dir"] for row in rows) + "\n", encoding="utf-8")

    print(f"completed_runs={len(rows)}")
    print(f"summary={args.output}")
    print(f"runs_list={runs_list_output}")


if __name__ == "__main__":
    main()
