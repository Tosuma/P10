from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import load_config
from src.utils import ensure_dir, read_csv_rows, resolve_seed, write_csv_rows, write_json


def fixed_grid_patches(image_width: int, image_height: int, patch_width: int, patch_height: int) -> list[tuple[int, int, int, int]]:
    coords = []
    for top in range(0, image_height, patch_height):
        for left in range(0, image_width, patch_width):
            width = min(patch_width, image_width - left)
            height = min(patch_height, image_height - top)
            coords.append((left, top, width, height))
    return coords


def random_crop_patches(
    image_width: int,
    image_height: int,
    patch_width: int,
    patch_height: int,
    count: int,
    seed: int,
) -> list[tuple[int, int, int, int]]:
    rng = random.Random(seed)
    max_left = max(image_width - patch_width, 0)
    max_top = max(image_height - patch_height, 0)
    patches = []
    for _ in range(count):
        left = rng.randint(0, max_left) if max_left > 0 else 0
        top = rng.randint(0, max_top) if max_top > 0 else 0
        patches.append((left, top, min(patch_width, image_width), min(patch_height, image_height)))
    return patches


def mask_crop_sum(mask_path: str, left: int, top: int, width: int, height: int) -> int:
    mask = np.asarray(Image.open(mask_path).crop((left, top, left + width, top + height)))
    return int((mask > 0).sum())


def build_patch_rows(rows: list[dict[str, str]], patch_cfg: dict, split: str, base_seed: int) -> list[dict[str, str]]:
    patch_width, patch_height = patch_cfg["size"]
    patch_rows: list[dict[str, str]] = []
    keep_empty_probability = float(patch_cfg.get("keep_empty_probability", 0.25))
    patches_per_image = int(patch_cfg.get("patches_per_image", 8))

    for row_idx, row in enumerate(rows):
        with Image.open(row["mask_path"]) as mask_image:
            image_width, image_height = mask_image.size

        if patch_cfg["mode"] == "fixed_grid":
            candidates = fixed_grid_patches(image_width, image_height, patch_width, patch_height)
        else:
            candidates = random_crop_patches(
                image_width,
                image_height,
                patch_width,
                patch_height,
                count=patches_per_image,
                seed=base_seed + row_idx,
            )

        rng = random.Random(base_seed + row_idx)
        kept_any_positive = False
        crop_cache: dict[tuple[int, int, int, int], int] = {}
        for patch_index, (left, top, width, height) in enumerate(candidates):
            key = (left, top, width, height)
            weed_pixels = crop_cache.setdefault(key, mask_crop_sum(row["mask_path"], left, top, width, height))
            is_empty = int(weed_pixels == 0)
            should_keep = weed_pixels > 0 or rng.random() <= keep_empty_probability
            if not should_keep:
                continue
            kept_any_positive = kept_any_positive or weed_pixels > 0
            patch_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "split": split,
                    "patch_id": f"{row['sample_id']}__p{patch_index:03d}",
                    "origin": row["sample_id"],
                    "x": left,
                    "y": top,
                    "width": width,
                    "height": height,
                    "weed_pixels": weed_pixels,
                    "is_empty": is_empty,
                }
            )

        if split == "train" and not kept_any_positive and candidates:
            best_patch = max(
                enumerate(candidates),
                key=lambda item: crop_cache.setdefault(item[1], mask_crop_sum(row["mask_path"], *item[1])),
            )
            patch_index, (left, top, width, height) = best_patch
            weed_pixels = crop_cache[(left, top, width, height)]
            patch_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "split": split,
                    "patch_id": f"{row['sample_id']}__fallback_{patch_index:03d}",
                    "origin": row["sample_id"],
                    "x": left,
                    "y": top,
                    "width": width,
                    "height": height,
                    "weed_pixels": weed_pixels,
                    "is_empty": int(weed_pixels == 0),
                }
            )

    return patch_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic patch manifests from split manifests.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    config["seed"] = resolve_seed(config.get("seed"))
    patch_cfg = config["patch"]
    output_dir = ensure_dir(args.output_dir or config["paths"]["patch_manifest_dir"])

    summary = {"config": args.config, "splits": {}}
    fieldnames = ["sample_id", "split", "patch_id", "origin", "x", "y", "width", "height", "weed_pixels", "is_empty"]
    split_offsets = {"train": 0, "val": 100000, "test": 200000}
    for split in ("train", "val", "test"):
        rows = read_csv_rows(config["paths"][f"{split}_manifest"])
        patch_rows = build_patch_rows(rows, patch_cfg, split=split, base_seed=int(config["seed"]) + split_offsets[split])
        write_csv_rows(output_dir / f"{split}_patches.csv", patch_rows, fieldnames)
        summary["splits"][split] = {
            "samples": len(rows),
            "patches": len(patch_rows),
            "positive_patches": sum(int(row["weed_pixels"]) > 0 for row in patch_rows),
        }
    write_json(output_dir / "patch_summary.json", summary)


if __name__ == "__main__":
    main()
