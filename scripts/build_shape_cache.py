#!/usr/bin/env python3
"""
Pre-build the image shape cache for fast training startup.

Run once before your first training job:
    python scripts/build_shape_cache.py <rgb_dir> <ms_dir>

After this, train_mae.sh starts in seconds instead of 13+ minutes,
because _build_grid_index() reads from .shape_cache.json on Ceph
instead of opening rasterio for every image.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.patch_dataset import (
    AgriculturalPatchDataset,
    discover_image_records,
)


def main() -> None:
    rgb_dir = sys.argv[1] if len(sys.argv) > 1 else "data/RGB"
    ms_dir = sys.argv[2] if len(sys.argv) > 2 else "data/Multispectral"

    print(f"rgb_dir : {rgb_dir}")
    print(f"ms_dir  : {ms_dir}")

    t0 = time.time()
    records = discover_image_records(rgb_dir, ms_dir)
    print(f"Found {len(records)} image pairs — building shape cache...")

    # Instantiating in val mode triggers _build_grid_index(), which
    # opens rasterio for each image and saves the result to
    # {ms_dir}/.shape_cache.json for all future runs.
    ds = AgriculturalPatchDataset(records=records, mode="val")

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s — {len(ds)} val patches indexed.")
    print(f"Cache saved to: {ms_dir}/.shape_cache.json")
    print("Future training jobs will skip this step entirely.")


if __name__ == "__main__":
    main()