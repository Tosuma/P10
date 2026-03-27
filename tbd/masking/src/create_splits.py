from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from src.utils import ensure_dir, write_csv_rows, write_json

MSI_SUFFIXES = ("G", "R", "RE", "NIR")


def sample_id_from_rgb(path: Path) -> str:
    return path.stem


def sample_id_from_msi(path: Path) -> str:
    stem = path.stem
    for suffix in MSI_SUFFIXES:
        marker = f"_{suffix}"
        if stem.endswith(marker):
            return stem[: -len(marker)]
    return stem


def infer_group_id(sample_id: str, strategy: str) -> str:
    if strategy == "date":
        parts = sample_id.split("_")
        if len(parts) >= 5:
            return "_".join(parts[2:5])
    if strategy == "datetime":
        parts = sample_id.split("_")
        if len(parts) >= 6:
            return "_".join(parts[2:6])
    return sample_id


def build_rows(dataset_root: Path, synthetic_root: Path | None, synthetic_ext: str) -> list[dict[str, str]]:
    rgb_dir = dataset_root / "RGB"
    mask_dir = dataset_root / "Masks"
    msi_dir = dataset_root / "Multispectral"

    msi_map: dict[str, list[str]] = defaultdict(list)
    for tif_path in sorted(msi_dir.glob("*.TIF")):
        msi_map[sample_id_from_msi(tif_path)].append(str(tif_path.resolve()))

    synthetic_map: dict[str, str] = {}
    if synthetic_root is not None and synthetic_root.exists():
        for synthetic_path in sorted(synthetic_root.rglob(f"*{synthetic_ext}")):
            synthetic_map[synthetic_path.stem] = str(synthetic_path.resolve())

    rows: list[dict[str, str]] = []
    for rgb_path in sorted(rgb_dir.glob("*.JPG")):
        sample_id = sample_id_from_rgb(rgb_path)
        mask_path = mask_dir / f"{sample_id}.png"
        if not mask_path.exists():
            continue
        synthetic_path = synthetic_map.get(sample_id, "")
        rows.append(
            {
                "sample_id": sample_id,
                "rgb_path": str(rgb_path.resolve()),
                "synthetic_msi_path": synthetic_path,
                "real_msi_path": json.dumps(sorted(msi_map.get(sample_id, []))),
                "mask_path": str(mask_path.resolve()),
            }
        )
    return rows


def assign_splits(rows: list[dict[str, str]], seed: int, train_ratio: float, val_ratio: float, group_strategy: str) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        group_id = infer_group_id(row["sample_id"], group_strategy)
        row["group_id"] = group_id
        grouped[group_id].append(row)

    group_ids = list(grouped)
    random.Random(seed).shuffle(group_ids)
    total = len(group_ids)
    if total >= 3:
        train_count = max(1, int(round(total * train_ratio)))
        val_count = max(1, int(round(total * val_ratio)))
        if train_count + val_count >= total:
            overflow = train_count + val_count - (total - 1)
            train_count = max(1, train_count - overflow)
        test_start = train_count + val_count
    else:
        train_count = max(1, total - 1)
        val_count = max(0, total - train_count)
        test_start = total

    train_ids = set(group_ids[:train_count])
    val_ids = set(group_ids[train_count:test_start])

    split_rows: list[dict[str, str]] = []
    for row in rows:
        split = "test"
        if row["group_id"] in train_ids:
            split = "train"
        elif row["group_id"] in val_ids:
            split = "val"
        row = dict(row)
        row["split"] = split
        split_rows.append(row)
    return split_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Create aligned train/val/test split manifests.")
    parser.add_argument("--dataset-root", default="data/weedy-rice")
    parser.add_argument("--output-dir", default="data/splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--group-strategy", choices=["sample", "date", "datetime"], default="datetime")
    parser.add_argument("--synthetic-root", default=None)
    parser.add_argument("--synthetic-ext", default=".npy")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if args.synthetic_root:
        synthetic_root = Path(args.synthetic_root)
    else:
        default_synthetic_root = dataset_root / "Synthetic"
        synthetic_root = default_synthetic_root if default_synthetic_root.exists() else None
    out_dir = ensure_dir(args.output_dir)

    rows = build_rows(dataset_root, synthetic_root, args.synthetic_ext)
    rows = assign_splits(rows, args.seed, args.train_ratio, args.val_ratio, args.group_strategy)

    fieldnames = ["sample_id", "rgb_path", "synthetic_msi_path", "real_msi_path", "mask_path", "group_id", "split"]
    write_csv_rows(out_dir / "all_samples.csv", rows, fieldnames)
    for split in ("train", "val", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        write_csv_rows(out_dir / f"{split}.csv", split_rows, fieldnames)

    summary = {
        "dataset_root": str(dataset_root.resolve()),
        "synthetic_root": str(synthetic_root.resolve()) if synthetic_root else None,
        "seed": args.seed,
        "group_strategy": args.group_strategy,
        "counts": {split: sum(row["split"] == split for row in rows) for split in ("train", "val", "test")},
        "groups": {split: len({row["group_id"] for row in rows if row["split"] == split}) for split in ("train", "val", "test")},
    }
    write_json(out_dir / "split_summary.json", summary)


if __name__ == "__main__":
    main()
