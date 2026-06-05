#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from read_manifest import load_manifest


def build_tasks(
    configs: list[str],
    manifest_path: Path | None,
    repeats: int,
    base_seed: int,
    group_name: str,
) -> list[dict[str, str | int]]:
    if manifest_path is not None:
        source_tasks = load_manifest(manifest_path)
        if not source_tasks:
            raise SystemExit(f"{manifest_path}: manifest contains no tasks.")
        for index, task in enumerate(source_tasks):
            if task["kind"] not in {"train", "baseline"}:
                raise SystemExit(
                    f"{manifest_path}: task {index} has kind={task['kind']!r}; multi-seed Slurm runs support train and baseline tasks only."
                )
    else:
        if not configs:
            raise SystemExit("At least one --config or one --manifest is required.")
        source_tasks = [
            {
                "group": group_name,
                "kind": "train",
                "config": config,
                "split": "test",
                "seed": "",
            }
            for config in configs
        ]

    tasks: list[dict[str, str | int]] = []
    for repeat_index in range(repeats):
        for task in source_tasks:
            source_group = str(task["group"])
            expanded_group = source_group if source_group.endswith("_multi_seed") else f"{source_group}_multi_seed"
            tasks.append(
                {
                    "group": expanded_group,
                    "kind": str(task["kind"]),
                    "config": str(task["config"]),
                    "split": str(task.get("split", "test") or "test"),
                    "seed": base_seed + repeat_index,
                }
            )
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand Slurm train or baseline tasks into one task per seed.")
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--group-name", default="multi_seed_train")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1.")

    tasks = build_tasks(args.config, args.manifest, args.repeats, args.base_seed, args.group_name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"tasks": tasks}
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"wrote={args.output}")
    print(f"tasks={len(tasks)}")
    if tasks:
        print(f"first_seed={tasks[0]['seed']}")
        print(f"last_seed={tasks[-1]['seed']}")


if __name__ == "__main__":
    main()
