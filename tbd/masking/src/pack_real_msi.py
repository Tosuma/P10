from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from src.create_splits import MSI_SUFFIXES, sample_id_from_msi
from src.utils import ensure_dir, write_json


def load_band(path: Path) -> np.ndarray:
    array = np.asarray(Image.open(path), dtype=np.float32)
    if array.ndim == 3:
        array = array[..., 0]
    return array


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack real multispectral band TIFFs into single .npy files.")
    parser.add_argument("--input-dir", default="data/weedy-rice/Multispectral")
    parser.add_argument("--output-dir", default="data/weedy-rice/MultispectralNPY")
    parser.add_argument("--input-ext", default=".TIF")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = ensure_dir(args.output_dir)

    grouped_paths: dict[str, dict[str, Path]] = defaultdict(dict)
    for tif_path in sorted(input_dir.glob(f"*{args.input_ext}")):
        sample_id = sample_id_from_msi(tif_path)
        suffix = tif_path.stem[len(sample_id) + 1 :]
        grouped_paths[sample_id][suffix] = tif_path

    packed = 0
    for i, (sample_id, band_map) in enumerate(grouped_paths.items()):
        if (output_dir / f"{sample_id}.npy").exists():
            continue
        
        missing = [band for band in MSI_SUFFIXES if band not in band_map]
        if missing:
            raise RuntimeError(f"Missing bands for {sample_id}: {missing}")

        print(f"Packing: {i+1}")
        stacked = np.stack([load_band(band_map[band]) for band in MSI_SUFFIXES], axis=-1).astype(np.float32)
        np.save(output_dir / f"{sample_id}.npy", stacked)
        packed += 1

    write_json(
        output_dir / "pack_summary.json",
        {
            "input_dir": str(input_dir.resolve()),
            "output_dir": str(Path(output_dir).resolve()),
            "bands": list(MSI_SUFFIXES),
            "packed_samples": packed,
        },
    )


if __name__ == "__main__":
    main()
