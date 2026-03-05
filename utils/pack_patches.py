"""
Pack pre-extracted patches from separate per-modality files into a single
.npz archive per patch.

Before packing, each sample requires 5 separate file opens on the network
filesystem (1 JPEG + 4 .npy files).  Ceph (and similar network filesystems)
incur a metadata round-trip per open, so with large batches and many DataLoader
workers the metadata server becomes the bottleneck.

After packing, each sample requires exactly 1 file open, cutting metadata
calls by 80% and recovering ~2-3× GPU utilisation during training.

Packed layout:
    WeedyRice-patches/
        Packed/
            {stem}.npz   → keys: rgb (H,W,3 uint8), ms (H,W,4 float32),
                                  ndvi (H,W float32), ndre (H,W float32),
                                  savi (H,W float32)

Usage:
    python utils/pack_patches.py --patch-dir data/WeedyRice-patches --workers 8

SLURM: sbatch scripts/slurm/pack_patches.sh
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------

def _pack_one(args: tuple[Path, str]) -> str | None:
    """Pack a single patch.  Returns stem on success, None on error."""
    patch_dir, stem = args
    out_path = patch_dir / "Packed" / f"{stem}.npz"
    if out_path.exists():
        return stem  # already packed

    try:
        rgb = cv2.imread(str(patch_dir / "RGB" / f"{stem}.jpg"))
        if rgb is None:
            return None
        ms   = np.load(patch_dir / "Multispectral" / f"{stem}.npy")
        ndvi = np.load(patch_dir / "NDVI"          / f"{stem}.npy")
        ndre = np.load(patch_dir / "NDRE"          / f"{stem}.npy")
        savi = np.load(patch_dir / "SAVI"          / f"{stem}.npy")
    except Exception as e:
        print(f"[WARN] {stem}: {e}")
        return None

    np.savez(out_path, rgb=rgb, ms=ms, ndvi=ndvi, ndre=ndre, savi=savi)
    return stem


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack patches into single .npz files.")
    parser.add_argument("--patch-dir", required=True, type=Path)
    parser.add_argument("--workers",   type=int, default=8)
    args = parser.parse_args()

    patch_dir = Path(args.patch_dir)
    out_dir   = patch_dir / "Packed"
    out_dir.mkdir(exist_ok=True)

    stems = sorted(p.stem for p in (patch_dir / "RGB").glob("*.jpg"))
    print(f"Packing {len(stems):,} patches → {out_dir}")

    tasks = [(patch_dir, s) for s in stems]
    done  = 0
    errors = 0
    with mp.Pool(args.workers) as pool:
        for result in pool.imap_unordered(_pack_one, tasks, chunksize=64):
            done += 1
            if result is None:
                errors += 1
            if done % 5000 == 0:
                print(f"  {done:,}/{len(stems):,}  errors={errors}")

    print(f"Done. {done - errors:,} packed, {errors} errors.")


if __name__ == "__main__":
    main()