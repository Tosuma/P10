"""
Pack all patches into a single memory-mapped numpy file on Ceph.

Each worker opens the memmap once at Dataset init, then each sample access
is a single seek+read with NO per-sample file opens.  Ceph serves large
sequential block reads efficiently, eliminating the metadata bottleneck.

Output: <patch_dir>/memmap/data.npy   shape (N, 10, 128, 128) float32
        <patch_dir>/memmap/stems.json  ordered list of patch stems

Usage:
    python utils/build_memmap.py --patch-dir data/WeedyRice-patches --workers 8

SLURM: sbatch scripts/slurm/build_memmap.sh
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np

CHANNELS = 10
PATCH    = 128


def _load_patch(args: tuple[Path, str]) -> np.ndarray | None:
    """Load and normalise one patch.  Returns (10, 128, 128) float32 or None."""
    patch_dir, stem = args
    try:
        # Check for packed npz first (faster if pack_patches was already run)
        npz_path = patch_dir / "Packed" / f"{stem}.npz"
        if npz_path.exists():
            d    = np.load(npz_path)
            rgb  = d["rgb"][:, :, ::-1].astype(np.float32) / 255.0   # BGR→RGB
            ms   = np.clip(d["ms"].astype(np.float32) / 65535.0, 0, 1)
            ndvi = (np.clip(d["ndvi"].astype(np.float32), -1, 1) + 1) / 2
            ndre = (np.clip(d["ndre"].astype(np.float32), -1, 1) + 1) / 2
            savi = (np.clip(d["savi"].astype(np.float32), -1.5, 1.5) + 1.5) / 3
        else:
            rgb_bgr = cv2.imread(str(patch_dir / "RGB" / f"{stem}.jpg"))
            if rgb_bgr is None:
                return None
            rgb  = rgb_bgr[:, :, ::-1].astype(np.float32) / 255.0
            ms   = np.clip(np.load(patch_dir / "Multispectral" / f"{stem}.npy").astype(np.float32) / 65535.0, 0, 1)
            ndvi = (np.clip(np.load(patch_dir / "NDVI" / f"{stem}.npy").astype(np.float32), -1, 1) + 1) / 2
            ndre = (np.clip(np.load(patch_dir / "NDRE" / f"{stem}.npy").astype(np.float32), -1, 1) + 1) / 2
            savi = (np.clip(np.load(patch_dir / "SAVI" / f"{stem}.npy").astype(np.float32), -1.5, 1.5) + 1.5) / 3

        img = np.concatenate(
            [rgb.transpose(2, 0, 1), ms.transpose(2, 0, 1),
             ndvi[None], ndre[None], savi[None]], axis=0
        ).astype(np.float32)  # (10, 128, 128)
        return img
    except Exception as e:
        print(f"[WARN] {stem}: {e}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-dir", required=True, type=Path)
    parser.add_argument("--workers",   type=int, default=8)
    args = parser.parse_args()

    patch_dir = Path(args.patch_dir)
    out_dir   = patch_dir / "memmap"
    out_dir.mkdir(exist_ok=True)

    stems = sorted(p.stem for p in (patch_dir / "RGB").glob("*.jpg"))
    N     = len(stems)
    print(f"Building memmap for {N:,} patches → {out_dir}/data.npy")
    print(f"File size: {N * CHANNELS * PATCH * PATCH * 4 / 1e9:.1f} GB")

    # Create the memmap file up-front
    mm = np.memmap(
        out_dir / "data.npy",
        dtype="float32",
        mode="w+",
        shape=(N, CHANNELS, PATCH, PATCH),
    )

    tasks = [(patch_dir, s) for s in stems]
    valid_stems: list[str] = []
    write_idx = 0
    errors    = 0

    with mp.Pool(args.workers) as pool:
        for idx, result in enumerate(pool.imap(_load_patch, tasks, chunksize=64)):
            if result is not None:
                mm[write_idx] = result
                valid_stems.append(stems[idx])
                write_idx += 1
            else:
                errors += 1
            if (idx + 1) % 10000 == 0:
                mm.flush()
                print(f"  {idx+1:,}/{N:,}  written={write_idx:,}  errors={errors}")

    mm.flush()
    del mm

    # Trim to valid count if any errors occurred
    if errors > 0:
        print(f"Trimming memmap to {write_idx:,} valid patches (skipped {errors}).")
        src = np.memmap(out_dir / "data.npy", dtype="float32", mode="r",
                        shape=(N, CHANNELS, PATCH, PATCH))
        dst = np.memmap(out_dir / "data_trimmed.npy", dtype="float32", mode="w+",
                        shape=(write_idx, CHANNELS, PATCH, PATCH))
        dst[:] = src[:write_idx]
        dst.flush()
        del src, dst
        (out_dir / "data.npy").unlink()
        (out_dir / "data_trimmed.npy").rename(out_dir / "data.npy")

    with open(out_dir / "stems.json", "w") as f:
        json.dump(valid_stems, f)

    print(f"Done. {write_idx:,} patches written, {errors} errors.")


if __name__ == "__main__":
    main()
