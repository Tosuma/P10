#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from json import JSONDecodeError
from pathlib import Path


def load_manifest(path: Path) -> list[dict[str, str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except JSONDecodeError as exc:
        raise SystemExit(f"{path}: invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc
    if isinstance(payload, dict):
        tasks = payload.get("tasks")
    elif isinstance(payload, list):
        tasks = payload
    else:
        tasks = None

    if not isinstance(tasks, list):
        raise SystemExit(f"{path}: expected a top-level 'tasks' list.")

    normalized = []
    required = ("group", "kind", "config")
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise SystemExit(f"{path}: task {index} must be a mapping.")
        missing = [key for key in required if not task.get(key)]
        if missing:
            raise SystemExit(f"{path}: task {index} is missing {', '.join(missing)}.")

        row = {
            "group": str(task["group"]),
            "kind": str(task["kind"]),
            "config": str(task["config"]),
            "split": str(task.get("split", "test")),
        }
        if any("\t" in value or "\n" in value for value in row.values()):
            raise SystemExit(f"{path}: task {index} contains unsupported tab or newline characters.")
        normalized.append(row)

    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Read a masking Slurm JSON manifest.")
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()

    for task in load_manifest(args.manifest):
        print("\t".join([task["group"], task["kind"], task["config"], task["split"]]))


if __name__ == "__main__":
    main()
