from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config
from src.utils import read_json, write_csv_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize modality experiment metrics into one comparison table.")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = []
    for run_dir in args.runs:
        run_path = Path(run_dir)
        config = load_config(run_path / "config.yaml")
        metadata = read_json(run_path / "run_metadata.json")
        eval_metrics = read_json(run_path / "evaluation" / "test" / "overall_metrics.json")
        best_summary = read_json(run_path / "metrics" / "best_summary.json")
        rows.append(
            {
                "experiment_name": config["experiment_name"],
                "modality": config["modality"],
                "seed": metadata["seed"],
                "val_iou": best_summary["best_val_iou"],
                "test_iou": eval_metrics["image_level"]["mean_iou"],
                "test_dice": eval_metrics["image_level"]["mean_dice"],
                "precision": eval_metrics["image_level"]["mean_precision"],
                "recall": eval_metrics["image_level"]["mean_recall"],
            }
        )
    write_csv_rows(args.output, rows, list(rows[0].keys()) if rows else ["experiment_name"])


if __name__ == "__main__":
    main()
