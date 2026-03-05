#!/usr/bin/env python3
"""
Compute per-image NDVI, NDRE, and SAVI maps from multispectral TIF bands
and write them as float32 GeoTIFFs into sibling folders of the input data.

Output layout (sibling folders inside DATA_ROOT)
-------------------------------------------------
DATA_ROOT/
├── Multispectral/   (input — existing)
├── NDVI/            {stem}.tif   float32, range ≈ [-1, 1]
├── NDRE/            {stem}.tif   float32, range ≈ [-1, 1]
└── SAVI/            {stem}.tif   float32, range ≈ [-1.5, 1.5]  (L = 0.5)

Formulas
--------
  NDVI  = (NIR − R)  / (NIR + R  + ε)
  NDRE  = (NIR − RE) / (NIR + RE + ε)
  SAVI  = (NIR − R)  / (NIR + R  + L + ε) × (1 + L),   L = 0.5

Usage
-----
python utils/compute_vegetation_indices.py \
    --data-root /path/to/WeedyRice-RGBMS-DB \
    --workers   8
"""

import argparse
import logging
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.transform import Affine
from tqdm import tqdm

# ── Constants ────────────────────────────────────────────────────────────────

SAVI_L: float = 0.5          # soil-brightness correction factor
EPS: float = 1e-7            # numerical stability for ratio denominators

# ── Per-image worker ─────────────────────────────────────────────────────────


def _compute_and_save(task: dict) -> Tuple[str, Optional[str]]:
    """Read R / RE / NIR bands, compute NDVI, NDRE, SAVI, save TIFs.

    Returns (stem, error_message_or_None).
    """
    stem: str = task["stem"]
    r_path: Path = task["r_path"]
    re_path: Path = task["re_path"]
    nir_path: Path = task["nir_path"]
    ndvi_out: Path = task["ndvi_out"]
    ndre_out: Path = task["ndre_out"]
    savi_out: Path = task["savi_out"]
    skip_existing: bool = task["skip_existing"]

    if skip_existing and ndvi_out.exists() and ndre_out.exists() and savi_out.exists():
        return stem, None

    try:
        # ── Load bands ───────────────────────────────────────────────────────
        with rasterio.open(str(nir_path)) as src:
            profile = src.profile.copy()
            nir = src.read(1).astype(np.float32)

        with rasterio.open(str(r_path)) as src:
            r = src.read(1).astype(np.float32)

        with rasterio.open(str(re_path)) as src:
            re = src.read(1).astype(np.float32)

        # ── Compute indices ──────────────────────────────────────────────────
        ndvi = (nir - r) / (nir + r + EPS)
        ndre = (nir - re) / (nir + re + EPS)
        savi = ((nir - r) / (nir + r + SAVI_L + EPS)) * (1.0 + SAVI_L)

        # ── Prepare output profile ───────────────────────────────────────────
        out_profile = profile.copy()
        out_profile.update(
            dtype=rasterio.float32,
            count=1,
            compress="deflate",
        )

        # ── Write TIFs ───────────────────────────────────────────────────────
        for array, path in [(ndvi, ndvi_out), (ndre, ndre_out), (savi, savi_out)]:
            with rasterio.open(str(path), "w", **out_profile) as dst:
                dst.write(array, 1)

        return stem, None

    except Exception as exc:  # noqa: BLE001
        return stem, str(exc)


# ── Task builder ─────────────────────────────────────────────────────────────


def _build_tasks(data_root: Path, skip_existing: bool) -> list[dict]:
    ms_dir = data_root / "Multispectral"
    ndvi_dir = data_root / "NDVI"
    ndre_dir = data_root / "NDRE"
    savi_dir = data_root / "SAVI"

    for d in (ndvi_dir, ndre_dir, savi_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Discover stems via NIR files — one per image
    nir_files = sorted(ms_dir.glob("*_NIR.TIF"))
    if not nir_files:
        raise RuntimeError(f"No *_NIR.TIF files found in {ms_dir}")

    tasks = []
    missing = 0
    for nir_path in nir_files:
        stem = nir_path.stem[: -len("_NIR")]  # strip trailing _NIR
        r_path = ms_dir / f"{stem}_R.TIF"
        re_path = ms_dir / f"{stem}_RE.TIF"

        absent = [p for p in (r_path, re_path) if not p.exists()]
        if absent:
            logging.warning(
                "Skipping '%s': missing %s",
                stem,
                ", ".join(str(p) for p in absent),
            )
            missing += 1
            continue

        tasks.append(
            {
                "stem": stem,
                "r_path": r_path,
                "re_path": re_path,
                "nir_path": nir_path,
                "ndvi_out": ndvi_dir / f"{stem}.tif",
                "ndre_out": ndre_dir / f"{stem}.tif",
                "savi_out": savi_dir / f"{stem}.tif",
                "skip_existing": skip_existing,
            }
        )

    if missing:
        logging.warning("Skipped %d image(s) due to missing bands.", missing)

    return tasks


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute NDVI, NDRE, SAVI maps from WeedyRice multispectral TIFs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/WeedyRice-RGBMS-DB"),
        help="Root containing the Multispectral/ folder.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, mp.cpu_count() // 2),
        help="Number of parallel worker processes.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip images whose three output TIFs already exist.",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Recompute even if outputs already exist.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    data_root: Path = args.data_root.expanduser().resolve()
    if not data_root.exists():
        logging.error("Data root not found: %s", data_root)
        sys.exit(1)

    logging.info("Data root : %s", data_root)
    logging.info("Workers   : %d", args.workers)

    tasks = _build_tasks(data_root, args.skip_existing)
    logging.info("Found %d images to process.", len(tasks))

    if not tasks:
        logging.info("Nothing to do.")
        return

    errors: list[tuple[str, str]] = []

    if args.workers == 1:
        for task in tqdm(tasks, desc="Computing VIs", unit="image"):
            stem, err = _compute_and_save(task)
            if err:
                errors.append((stem, err))
    else:
        with mp.Pool(processes=args.workers) as pool:
            for stem, err in tqdm(
                pool.imap_unordered(_compute_and_save, tasks, chunksize=8),
                total=len(tasks),
                desc="Computing VIs",
                unit="image",
            ):
                if err:
                    errors.append((stem, err))

    succeeded = len(tasks) - len(errors)
    logging.info("Done. %d / %d images processed successfully.", succeeded, len(tasks))

    if errors:
        logging.error("%d image(s) failed:", len(errors))
        for stem, err in errors:
            logging.error("  [%s] %s", stem, err)
        sys.exit(1)


if __name__ == "__main__":
    main()