#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from json import JSONDecodeError
from pathlib import Path

EMPTY_FIELD = "__MASKING_EMPTY__"


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
        if task.get("resume_checkpoint") and task.get("resume_run_dir"):
            raise SystemExit(f"{path}: task {index} must not set both resume_checkpoint and resume_run_dir.")

        resume_checkpoint = task.get("resume_checkpoint")
        if task.get("resume_run_dir"):
            resume_run_dir = str(task["resume_run_dir"]).rstrip("/\\")
            resume_checkpoint = f"{resume_run_dir}/checkpoints/latest.pt"
        if resume_checkpoint and str(task["kind"]) != "train":
            raise SystemExit(f"{path}: task {index} can only set resume_checkpoint/resume_run_dir for kind='train'.")

        row = {
            "group": str(task["group"]),
            "kind": str(task["kind"]),
            "config": str(task["config"]),
            "split": str(task.get("split", "test")),
            "seed": "" if task.get("seed") is None else str(task["seed"]),
            "resume_checkpoint": "" if resume_checkpoint is None else str(resume_checkpoint),
        }
        if any("\t" in value or "\n" in value for value in row.values()):
            raise SystemExit(f"{path}: task {index} contains unsupported tab or newline characters.")
        if any(value == EMPTY_FIELD for value in row.values()):
            raise SystemExit(f"{path}: task {index} contains reserved value {EMPTY_FIELD!r}.")
        normalized.append(row)

    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Read a masking Slurm JSON manifest.")
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()

    for task in load_manifest(args.manifest):
        print(
            "\t".join(
                [
                    task["group"],
                    task["kind"],
                    task["config"],
                    task["split"],
                    task["seed"] or EMPTY_FIELD,
                    task["resume_checkpoint"] or EMPTY_FIELD,
                ]
            )
        )


if __name__ == "__main__":
    main()
