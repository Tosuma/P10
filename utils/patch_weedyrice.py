#!/usr/bin/env python3
"""
Patch the WeedyRice-RGBMS-DB dataset (including vegetation-index maps) into
128×128 tiles for fast retrieval during training.

All modalities are resized to a common resolution — the largest clean multiple
of PATCH_SIZE that fits within the native multispectral resolution (2592×1944).
With the default patch size of 128 this yields 2560×1920 (20 cols × 15 rows =
300 patches per image).

Prerequisites
-------------
Run ``utils/compute_vegetation_indices.py`` first so that the NDVI/, NDRE/ and
SAVI/ folders exist inside DATA_ROOT.  Pass ``--skip-vi`` to skip those three
modalities if they have not been computed yet.

Output layout
-------------
WeedyRice-patches/
├── RGB/           {stem}_r{row:03d}_c{col:03d}.jpg
├── Multispectral/ {stem}_r{row:03d}_c{col:03d}.npy  float32 H×W×4  (G/R/RE/NIR)
├── Masks/         {stem}_r{row:03d}_c{col:03d}.png  uint8           (0=bg, 255=weed)
├── Overlay/       {stem}_r{row:03d}_c{col:03d}.jpg
├── Synthetic/     {stem}_r{row:03d}_c{col:03d}.npy  uint8  H×W×4   (bands 1-4)
├── NDVI/          {stem}_r{row:03d}_c{col:03d}.npy  float32 H×W
├── NDRE/          {stem}_r{row:03d}_c{col:03d}.npy  float32 H×W
└── SAVI/          {stem}_r{row:03d}_c{col:03d}.npy  float32 H×W

Usage
-----
python utils/patch_weedyrice.py \
    --data-root   /path/to/WeedyRice-RGBMS-DB \
    --output-root /path/to/WeedyRice-patches \
    --patch-size  128 \
    --workers     16
"""

import argparse
import csv
import logging
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rasterio
from tqdm import tqdm

# ── Constants ────────────────────────────────────────────────────────────────

MS_BANDS: List[str] = ["G", "R", "RE", "NIR"]
SYN_BANDS: List[int] = [1, 2, 3, 4]
VI_NAMES: List[str] = ["NDVI", "NDRE", "SAVI"]

# Native multispectral resolution used to derive a common target resolution.
_NATIVE_MS_W = 2592
_NATIVE_MS_H = 1944

# ── Dimension helpers ────────────────────────────────────────────────────────


def _target_dims(patch_size: int) -> Tuple[int, int]:
    """Return (width, height) — the largest multiple of patch_size that fits
    within the native multispectral resolution."""
    return (
        (_NATIVE_MS_W // patch_size) * patch_size,
        (_NATIVE_MS_H // patch_size) * patch_size,
    )


# ── Loaders ──────────────────────────────────────────────────────────────────


def _load_bgr(path: Path, target_w: int, target_h: int) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def _load_gray(path: Path, target_w: int, target_h: int, nearest: bool = False) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read grayscale image: {path}")
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.resize(img, (target_w, target_h), interpolation=interp)


def _load_multispectral(band_paths: Dict[str, Path], target_w: int, target_h: int) -> np.ndarray:
    """Read 4 single-band TIFs → float32 H×W×4 (bands: G, R, RE, NIR)."""
    channels = []
    for band in MS_BANDS:
        with rasterio.open(str(band_paths[band])) as src:
            data = src.read(1).astype(np.float32)
        channels.append(cv2.resize(data, (target_w, target_h), interpolation=cv2.INTER_LINEAR))
    return np.stack(channels, axis=-1)


def _load_synthetic(band_paths: Dict[int, Path], target_w: int, target_h: int) -> np.ndarray:
    """Read 4 single-channel JPGs → uint8 H×W×4 (bands: 1, 2, 3, 4)."""
    channels = []
    for band in SYN_BANDS:
        channels.append(_load_gray(band_paths[band], target_w, target_h))
    return np.stack(channels, axis=-1)


def _load_vi(path: Path, target_w: int, target_h: int) -> np.ndarray:
    """Read a float32 vegetation-index GeoTIF → float32 H×W."""
    with rasterio.open(str(path)) as src:
        data = src.read(1).astype(np.float32)
    return cv2.resize(data, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


# ── Per-image worker ─────────────────────────────────────────────────────────


def _process_image(task: dict) -> Tuple[str, Optional[str]]:
    """Slice one image group into patches and write all outputs.

    Returns (stem, error_message_or_None).
    """
    stem: str = task["stem"]
    paths: dict = task["paths"]
    out_dirs: Dict[str, Path] = task["out_dirs"]
    patch_size: int = task["patch_size"]
    target_w: int = task["target_w"]
    target_h: int = task["target_h"]
    skip_existing: bool = task["skip_existing"]
    include_vi: bool = task["include_vi"]

    try:
        # ── Load all modalities ───────────────────────────────────────────────
        rgb = _load_bgr(paths["rgb"], target_w, target_h)
        mask = _load_gray(paths["mask"], target_w, target_h, nearest=True)
        overlay = _load_bgr(paths["overlay"], target_w, target_h)
        ms = _load_multispectral(paths["ms"], target_w, target_h)
        syn = _load_synthetic(paths["synthetic"], target_w, target_h)

        vi_arrays: Dict[str, np.ndarray] = {}
        if include_vi:
            for vi in VI_NAMES:
                vi_arrays[vi] = _load_vi(paths["vi"][vi], target_w, target_h)

        # ── Slice into patches ────────────────────────────────────────────────
        n_rows = target_h // patch_size
        n_cols = target_w // patch_size

        for r in range(n_rows):
            for c in range(n_cols):
                y0, y1 = r * patch_size, (r + 1) * patch_size
                x0, x1 = c * patch_size, (c + 1) * patch_size
                patch_stem = f"{stem}_r{r:03d}_c{c:03d}"

                rgb_out = out_dirs["RGB"] / f"{patch_stem}.jpg"
                if skip_existing and rgb_out.exists():
                    continue

                cv2.imwrite(str(rgb_out), rgb[y0:y1, x0:x1],
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(str(out_dirs["Masks"] / f"{patch_stem}.png"),
                            mask[y0:y1, x0:x1])
                cv2.imwrite(str(out_dirs["Overlay"] / f"{patch_stem}.jpg"),
                            overlay[y0:y1, x0:x1],
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                np.save(str(out_dirs["Multispectral"] / f"{patch_stem}.npy"),
                        ms[y0:y1, x0:x1])
                np.save(str(out_dirs["Synthetic"] / f"{patch_stem}.npy"),
                        syn[y0:y1, x0:x1])

                if include_vi:
                    for vi in VI_NAMES:
                        np.save(
                            str(out_dirs[vi] / f"{patch_stem}.npy"),
                            vi_arrays[vi][y0:y1, x0:x1],
                        )

        return stem, None

    except Exception as exc:  # noqa: BLE001
        return stem, str(exc)


# ── Task builder ─────────────────────────────────────────────────────────────


def _load_csv_id_map(data_root: Path) -> Dict[str, int]:
    """Try to read stem → synthetic-id from Metadata/filename_mapping.csv.
    Returns an empty dict if the file is missing or lacks the right columns.
    """
    csv_path = data_root / "Metadata" / "filename_mapping.csv"
    if not csv_path.exists():
        return {}

    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        id_col = next((f for f in fieldnames if f.lower() in ("id", "index")), None)
        name_col = next(
            (f for f in fieldnames if "name" in f.lower() or "file" in f.lower()), None
        )
        if not id_col or not name_col:
            return {}

        mapping: Dict[str, int] = {}
        for row in reader:
            try:
                mapping[Path(row[name_col]).stem] = int(row[id_col])
            except (ValueError, KeyError):
                pass
    return mapping


def _build_tasks(
    data_root: Path,
    out_dirs: Dict[str, Path],
    patch_size: int,
    target_w: int,
    target_h: int,
    skip_existing: bool,
    include_vi: bool,
) -> List[dict]:
    rgb_dir = data_root / "RGB"
    ms_dir = data_root / "Multispectral"
    mask_dir = data_root / "Masks"
    overlay_dir = data_root / "Overlay"
    syn_dir = data_root / "Synthetic"

    rgb_files = sorted(rgb_dir.glob("*.JPG"))
    if not rgb_files:
        raise RuntimeError(f"No .JPG files found in {rgb_dir}")

    # Synthetic ID mapping: prefer CSV, fall back to alphabetical sort index
    csv_map = _load_csv_id_map(data_root)
    stem_to_id: Dict[str, int] = (
        csv_map if csv_map else {f.stem: i for i, f in enumerate(rgb_files)}
    )

    tasks: List[dict] = []
    skipped = 0

    for rgb_file in rgb_files:
        stem = rgb_file.stem
        mask_path = mask_dir / f"{stem}.png"
        overlay_path = overlay_dir / f"{stem}.JPG"
        ms_paths = {band: ms_dir / f"{stem}_{band}.TIF" for band in MS_BANDS}

        syn_id = stem_to_id.get(stem)
        if syn_id is None:
            logging.warning("No synthetic ID for '%s'; skipping.", stem)
            skipped += 1
            continue
        syn_paths = {
            band: syn_dir / f"validation_result_{syn_id}_{band}_.JPG"
            for band in SYN_BANDS
        }

        vi_paths: Dict[str, Path] = {}
        if include_vi:
            for vi in VI_NAMES:
                vi_paths[vi] = data_root / vi / f"{stem}.tif"

        # Validate existence
        required = (
            [mask_path, overlay_path]
            + list(ms_paths.values())
            + list(syn_paths.values())
            + (list(vi_paths.values()) if include_vi else [])
        )
        missing = [p for p in required if not p.exists()]
        if missing:
            logging.warning(
                "Skipping '%s': missing:\n  %s",
                stem,
                "\n  ".join(str(p) for p in missing),
            )
            skipped += 1
            continue

        tasks.append(
            {
                "stem": stem,
                "paths": {
                    "rgb": rgb_file,
                    "mask": mask_path,
                    "overlay": overlay_path,
                    "ms": ms_paths,
                    "synthetic": syn_paths,
                    "vi": vi_paths,
                },
                "out_dirs": out_dirs,
                "patch_size": patch_size,
                "target_w": target_w,
                "target_h": target_h,
                "skip_existing": skip_existing,
                "include_vi": include_vi,
            }
        )

    if skipped:
        logging.warning("Skipped %d image(s) due to missing files.", skipped)

    return tasks


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch WeedyRice-RGBMS-DB (+ VI maps) into 128×128 tiles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/WeedyRice-RGBMS-DB"),
        help="Root of the WeedyRice-RGBMS-DB dataset (must contain RGB/, Multispectral/, …).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/WeedyRice-patches"),
        help="Destination for patch directories.",
    )
    parser.add_argument(
        "--patch-size", type=int, default=128, help="Patch side length in pixels."
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
        help="Skip image groups whose first RGB patch already exists (allows resume).",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Overwrite existing patches.",
    )
    parser.add_argument(
        "--skip-vi",
        action="store_true",
        default=False,
        help="Skip NDVI/NDRE/SAVI patching (use when VI maps have not been computed yet).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    data_root: Path = args.data_root.expanduser().resolve()
    output_root: Path = args.output_root.expanduser().resolve()
    patch_size: int = args.patch_size
    include_vi: bool = not args.skip_vi

    if not data_root.exists():
        logging.error("Data root not found: %s", data_root)
        sys.exit(1)

    # Warn early if VI dirs are absent but VI patching was requested
    if include_vi:
        missing_vi = [vi for vi in VI_NAMES if not (data_root / vi).exists()]
        if missing_vi:
            logging.error(
                "VI folder(s) not found: %s\n"
                "Run utils/compute_vegetation_indices.py first, or pass --skip-vi.",
                ", ".join(missing_vi),
            )
            sys.exit(1)

    target_w, target_h = _target_dims(patch_size)
    patches_per_image = (target_w // patch_size) * (target_h // patch_size)

    logging.info("Data root    : %s", data_root)
    logging.info("Output root  : %s", output_root)
    logging.info("Patch size   : %d px", patch_size)
    logging.info(
        "Target res   : %d × %d  (%d patches / image)", target_w, target_h, patches_per_image
    )
    logging.info("Workers      : %d", args.workers)
    logging.info("Include VI   : %s", include_vi)

    # Create output directories
    categories = ["RGB", "Multispectral", "Masks", "Overlay", "Synthetic"]
    if include_vi:
        categories += VI_NAMES
    out_dirs: Dict[str, Path] = {}
    for cat in categories:
        d = output_root / cat
        d.mkdir(parents=True, exist_ok=True)
        out_dirs[cat] = d

    logging.info("Scanning dataset …")
    tasks = _build_tasks(
        data_root, out_dirs, patch_size, target_w, target_h, args.skip_existing, include_vi
    )
    logging.info("Found %d image groups to process.", len(tasks))

    if not tasks:
        logging.info("Nothing to do.")
        return

    logging.info("Estimated output: ~%d patches per category.", len(tasks) * patches_per_image)

    errors: List[Tuple[str, str]] = []

    if args.workers == 1:
        for task in tqdm(tasks, desc="Patching", unit="image"):
            stem, err = _process_image(task)
            if err:
                errors.append((stem, err))
    else:
        with mp.Pool(processes=args.workers) as pool:
            for stem, err in tqdm(
                pool.imap_unordered(_process_image, tasks, chunksize=4),
                total=len(tasks),
                desc="Patching",
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