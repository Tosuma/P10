#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path


def parse_status(path: Path) -> dict[str, str]:
    payload: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in raw_line:
            continue
        key, value = raw_line.split("=", 1)
        payload[key] = value
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect successful run directories from Slurm task status files.")
    parser.add_argument("--status-dir", type=Path, required=True)
    args = parser.parse_args()

    if not args.status_dir.is_dir():
        raise SystemExit(f"Status directory was not found: {args.status_dir}")

    run_dirs: list[str] = []
    seen: set[str] = set()
    for status_path in sorted(args.status_dir.glob("task_*_attempt_*.status")):
        payload = parse_status(status_path)
        if payload.get("state") != "success":
            continue
        run_dir = payload.get("run_dir", "").strip()
        if not run_dir or run_dir in seen:
            continue
        seen.add(run_dir)
        run_dirs.append(run_dir)

    for run_dir in run_dirs:
        print(run_dir)


if __name__ == "__main__":
    main()
