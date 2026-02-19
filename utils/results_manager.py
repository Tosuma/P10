import json
import math
import argparse
from typing import Dict, List, Any
from pathlib import Path

import pandas as pd


FIELDS = [
    "MRAE",
    "MSE",
    "RMSE",
    "PSNR",
    "SSIM",
    "SAM",
    "NDVI_PRED",
    "NDVI_GT",
    "NDRE_PRED",
    "NDRE_GT",
    "MRAE_G",
    "MSE_G",
    "RMSE_G",
    "PSNR_G",
    "SSIM_G",
    "SAM_G",
    "MRAE_R",
    "MSE_R",
    "RMSE_R",
    "PSNR_R",
    "SSIM_R",
    "SAM_R",
    "MRAE_RE",
    "MSE_RE",
    "RMSE_RE",
    "PSNR_RE",
    "SSIM_RE",
    "SAM_RE",
    "MRAE_NIR",
    "MSE_NIR",
    "RMSE_NIR",
    "PSNR_NIR",
    "SSIM_NIR",
    "SAM_NIR",
]


def is_invalid_number(value: Any) -> bool:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return True
    return math.isnan(value) or math.isinf(value)


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data

    raise ValueError("Unsupported JSON structure")


def collect_fields(
    records: List[Dict[str, Any]],
    skip_invalid: bool = True,
) -> Dict[str, List[float]]:
    collected = {field: [] for field in FIELDS}

    for record in records:
        for field in FIELDS:
            if field not in record:
                continue

            value = record[field]
            if is_invalid_number(value):
                if not skip_invalid:
                    collected[field].append(float("nan"))
                continue

            collected[field].append(float(value))

    return collected


def median(values: List[float]) -> float:
    n = len(values)
    if n == 0:
        return float("nan")
    s = sorted(values)
    return s[n // 2]


def build_summary_row(
    metrics: Dict[str, List[float]],
    json_path: str,
) -> Dict[str, float]:
    row: Dict[str, float] = {
        "json_path": json_path,
        "num_records": max(len(v) for v in metrics.values() if v),
    }

    for field, values in metrics.items():
        if values:
            row[f"{field}_avg"] = sum(values) / len(values)
            row[f"{field}_median"] = median(values)
        else:
            row[f"{field}_avg"] = float("nan")
            row[f"{field}_median"] = float("nan")

    return row


def append_to_excel(
    row: Dict[str, float],
    excel_path: Path,
    sheet_name: str,
):
    df_new = pd.DataFrame([row])

    if excel_path.exists():
        df_existing = pd.read_excel(excel_path, sheet_name=sheet_name)
        df_out = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_out = df_new

    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="w") as writer:
        df_out.to_excel(writer, sheet_name=sheet_name, index=False)


def main():
    parser = argparse.ArgumentParser(description="Append JSON metrics to Excel")
    parser.add_argument("--json-path", required=True, help="Path to results.json")
    parser.add_argument("--excel-path", required=True, help="Path to output Excel file")
    parser.add_argument("--sheet", default="results", help="Excel sheet name")

    args = parser.parse_args()

    records = load_json(args.json_path)
    metrics = collect_fields(records, skip_invalid=True)
    row = build_summary_row(metrics, args.json_path)

    append_to_excel(
        row=row,
        excel_path=Path(args.excel_path),
        sheet_name=args.sheet,
    )


if __name__ == "__main__":
    main()
