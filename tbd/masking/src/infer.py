from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.datasets import normalize_image
from src.evaluate import load_checkpoint, load_preview_image, reconstruct_image_metrics
from src.utils import device_from_config, ensure_dir, read_csv_rows, read_json, write_json
from src.visualize import save_prediction_panel


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on a split manifest using a trained checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--patch-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    device = device_from_config()
    model, config, _ = load_checkpoint(args.checkpoint, device)
    output_dir = ensure_dir(args.output_dir)

    sample_rows = read_csv_rows(args.manifest)
    patch_rows = read_csv_rows(args.patch_manifest)
    sample_lookup = {row["sample_id"]: row for row in sample_rows}
    normalization = read_json(Path(config["paths"]["normalization_stats"]))
    threshold = float(config["evaluation"]["threshold"])

    predictions_by_sample = {}
    with torch.no_grad():
        for patch in patch_rows:
            sample = sample_lookup[patch["sample_id"]]
            image = load_preview_image(sample, config["modality"])
            left, top = int(patch["x"]), int(patch["y"])
            width, height = int(patch["width"]), int(patch["height"])
            crop = image[top : top + height, left : left + width, ...]
            crop = normalize_image(crop.astype(np.float32), config["modality"], normalization)
            crop_tensor = torch.from_numpy(np.moveaxis(crop, -1, 0)).unsqueeze(0).to(device)
            logits = model(crop_tensor)
            probability = torch.sigmoid(logits)[0, 0].cpu().numpy()
            predictions_by_sample.setdefault(sample["sample_id"], []).append({"coords": [left, top, width, height], "probability": probability})

    per_image_rows, visuals = reconstruct_image_metrics(sample_rows, predictions_by_sample, threshold)
    write_json(output_dir / "summary.json", {"samples": len(per_image_rows)})
    for visual in visuals:
        sample = sample_lookup[visual["sample_id"]]
        image = load_preview_image(sample, config["modality"])
        save_prediction_panel(
            Path(output_dir) / f"{visual['sample_id']}.png",
            image=image,
            gt_mask=visual["gt_mask"],
            probability_map=visual["probability_map"],
            pred_mask=visual["pred_mask"],
        )


if __name__ == "__main__":
    main()
