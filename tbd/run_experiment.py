"""
Run the full experiment: 3 variants × N seeds.

This is the main entry point. It trains all variants with identical
splits and hyperparameters, then produces a summary comparison.

Usage:
    python run_experiment.py --data_root ./data --seeds 42 123 456 789 1337

Output:
    results/summary.json — aggregated results for your thesis
    results/<variant>_seed<N>/ — individual run results and checkpoints
"""

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Run full segmentation experiment")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1337],
                        help="Random seeds for repeated runs (default: 5 seeds)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ms_channels", type=int, default=4,
                        help="Number of multispectral channels (G, R, RE, NIR)")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="./results")
    args = parser.parse_args()

    variants = ["rgb", "multispectral", "synthetic"]
    all_results = defaultdict(list)

    for seed in args.seeds:
        for variant in variants:
            print(f"\n{'='*60}")
            print(f"RUNNING: {variant} | seed={seed}")
            print(f"{'='*60}\n")

            train_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")
            cmd = [
                sys.executable, train_script,
                "--data_root", args.data_root,
                "--input_type", variant,
                "--seed", str(seed),
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--lr", str(args.lr),
                "--ms_channels", str(args.ms_channels),
                "--num_classes", str(args.num_classes),
                "--image_size", str(args.image_size),
                "--output_dir", args.output_dir,
            ]

            result = subprocess.run(cmd, capture_output=False)
            if result.returncode != 0:
                print(f"WARNING: Run failed for {variant} seed={seed}")
                continue

            # Load the results
            run_dir = os.path.join(args.output_dir, f"{variant}_seed{seed}")
            results_path = os.path.join(run_dir, "results.json")
            if os.path.exists(results_path):
                with open(results_path) as f:
                    run_results = json.load(f)
                all_results[variant].append(run_results)

    # ──────────── Summary ────────────
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    summary = {}
    for variant in variants:
        if variant not in all_results or not all_results[variant]:
            print(f"\n{variant}: NO RESULTS")
            continue

        mious = [r["test_miou"] for r in all_results[variant]]
        mean_miou = np.mean(mious)
        std_miou = np.std(mious)

        # Per-class IoU aggregation
        per_class_means = {}
        per_class_stds = {}
        class_names = list(all_results[variant][0]["test_per_class_iou"].keys())
        for cls in class_names:
            cls_ious = [r["test_per_class_iou"][cls] for r in all_results[variant]]
            # Filter NaN
            cls_ious = [x for x in cls_ious if not (isinstance(x, float) and np.isnan(x))]
            if cls_ious:
                per_class_means[cls] = float(np.mean(cls_ious))
                per_class_stds[cls] = float(np.std(cls_ious))

        summary[variant] = {
            "miou_mean": float(mean_miou),
            "miou_std": float(std_miou),
            "miou_all_seeds": mious,
            "per_class_iou_mean": per_class_means,
            "per_class_iou_std": per_class_stds,
            "num_runs": len(mious),
        }

        print(f"\n{variant.upper()}")
        print(f"  mIoU: {mean_miou:.4f} ± {std_miou:.4f}")
        for cls in class_names:
            if cls in per_class_means:
                print(f"  {cls} IoU: {per_class_means[cls]:.4f} ± {per_class_stds.get(cls, 0):.4f}")

    # Key comparison
    if "multispectral" in summary and "synthetic" in summary:
        diff = summary["multispectral"]["miou_mean"] - summary["synthetic"]["miou_mean"]
        print(f"\n{'─'*60}")
        print(f"Real MS vs Synthetic MS mIoU difference: {diff:+.4f}")
        if abs(diff) < 0.02:
            print("→ Difference < 2% — strong evidence of comparable performance")
        elif abs(diff) < 0.05:
            print("→ Difference < 5% — moderate evidence, synthetic data partially preserves information")
        else:
            print("→ Difference ≥ 5% — significant gap, investigate spectral reconstruction quality")

    if "synthetic" in summary and "rgb" in summary:
        diff = summary["synthetic"]["miou_mean"] - summary["rgb"]["miou_mean"]
        print(f"Synthetic MS vs RGB mIoU difference: {diff:+.4f}")
        if diff > 0.02:
            print("→ Synthetic multispectral adds value beyond RGB alone")
        elif diff > -0.02:
            print("→ Marginal difference — synthetic MS may not add significant value over RGB")
        else:
            print("→ RGB outperforms synthetic MS — reconstruction may introduce noise")

    # Save summary
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull summary saved to {summary_path}")


if __name__ == "__main__":
    main()
