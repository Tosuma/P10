#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


SUBMIT_RE = re.compile(
    r"Submitting task (?P<task_id>\d+) .*?: group=(?P<group>\S+) kind=(?P<kind>\S+) "
    r"config=(?P<config>\S+) split=(?P<split>\S+) seed=(?P<seed>\S+) "
    r"(?:resume_checkpoint=(?P<resume_checkpoint>\S+) )?attempt=(?P<attempt>\d+)"
)
SUCCESS_RE = re.compile(r"Task (?P<task_id>\d+) completed successfully")
PERMANENT_RE = re.compile(r"Task (?P<task_id>\d+): permanent failure after (?P<attempts>\d+) attempt")
STATUS_FAILED_RE = re.compile(
    r"Task (?P<task_id>\d+): status file reports state=(?P<state>\S+); "
    r"seed=(?P<seed>[^;]*); message=(?P<message>.*?); job_log=(?P<job_log>\S+)"
)
VALIDATION_FAILED_RE = re.compile(r"Task (?P<task_id>\d+): validation failed")
MISSING_RE = re.compile(r"Task (?P<task_id>\d+): (?P<message>missing .*|expected .*|transient .*)")


def read_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks = payload.get("tasks") if isinstance(payload, dict) else payload
    if not isinstance(tasks, list):
        raise SystemExit(f"{path}: expected a top-level 'tasks' list or a JSON task list.")
    normalized = []
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise SystemExit(f"{path}: task {index} must be an object.")
        resume_checkpoint = task.get("resume_checkpoint")
        if task.get("resume_run_dir"):
            resume_run_dir = str(task["resume_run_dir"]).rstrip("/\\")
            resume_checkpoint = f"{resume_run_dir}/checkpoints/latest.pt"
        normalized.append(
            {
                "task_id": index,
                "group": task.get("group", ""),
                "kind": task.get("kind", ""),
                "config": task.get("config", ""),
                "split": task.get("split", "test"),
                "seed": "" if task.get("seed") is None else str(task.get("seed")),
                "resume_checkpoint": "" if resume_checkpoint is None else str(resume_checkpoint),
            }
        )
    return normalized


def parse_controller_log(path: Path) -> dict[int, dict[str, Any]]:
    states: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "submissions": 0,
            "attempts": set(),
            "submitted_jobs": [],
            "success": False,
            "permanent_failure": False,
            "permanent_attempts": "",
            "failure_messages": [],
            "job_logs": [],
            "last_line": "",
        }
    )
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()

        match = SUBMIT_RE.search(line)
        if match:
            task_id = int(match.group("task_id"))
            state = states[task_id]
            state["submissions"] += 1
            state["attempts"].add(match.group("attempt"))
            state["last_line"] = line
            continue

        match = SUCCESS_RE.search(line)
        if match:
            task_id = int(match.group("task_id"))
            states[task_id]["success"] = True
            states[task_id]["last_line"] = line
            continue

        match = PERMANENT_RE.search(line)
        if match:
            task_id = int(match.group("task_id"))
            state = states[task_id]
            state["permanent_failure"] = True
            state["permanent_attempts"] = match.group("attempts")
            state["failure_messages"].append(line)
            state["last_line"] = line
            continue

        match = STATUS_FAILED_RE.search(line)
        if match:
            task_id = int(match.group("task_id"))
            state = states[task_id]
            state["failure_messages"].append(match.group("message"))
            state["job_logs"].append(match.group("job_log"))
            state["last_line"] = line
            continue

        match = VALIDATION_FAILED_RE.search(line)
        if match:
            task_id = int(match.group("task_id"))
            states[task_id]["failure_messages"].append(line)
            states[task_id]["last_line"] = line
            continue

        match = MISSING_RE.search(line)
        if match:
            task_id = int(match.group("task_id"))
            states[task_id]["failure_messages"].append(match.group("message"))
            states[task_id]["last_line"] = line

    return states


def classify(task: dict[str, Any], state: dict[str, Any] | None) -> dict[str, Any]:
    state = state or {}
    success = bool(state.get("success", False))
    permanent_failure = bool(state.get("permanent_failure", False))
    submissions = int(state.get("submissions", 0))

    if success:
        status = "success"
    elif permanent_failure:
        status = "permanent_failure"
    elif submissions > 0:
        status = "incomplete_or_interrupted"
    else:
        status = "never_submitted"

    messages = list(dict.fromkeys(state.get("failure_messages", [])))
    job_logs = list(dict.fromkeys(state.get("job_logs", [])))
    attempts = sorted(state.get("attempts", []), key=lambda value: int(value) if str(value).isdigit() else value)

    return {
        **task,
        "status": status,
        "submissions": submissions,
        "attempts": ",".join(str(attempt) for attempt in attempts),
        "permanent_attempts": state.get("permanent_attempts", ""),
        "failure_messages": " | ".join(messages),
        "job_logs": " | ".join(job_logs),
        "last_log_line": state.get("last_line", ""),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "task_id",
        "status",
        "group",
        "kind",
        "config",
        "split",
        "seed",
        "resume_checkpoint",
        "submissions",
        "attempts",
        "permanent_attempts",
        "failure_messages",
        "job_logs",
        "last_log_line",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_retry_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    tasks = []
    for row in rows:
        task: dict[str, Any] = {
            "group": row["group"],
            "kind": row["kind"],
            "config": row["config"],
            "split": row["split"],
        }
        if str(row.get("seed", "")) != "":
            task["seed"] = int(row["seed"]) if str(row["seed"]).isdigit() else row["seed"]
        if str(row.get("resume_checkpoint", "")) != "":
            task["resume_checkpoint"] = row["resume_checkpoint"]
        tasks.append(task)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"tasks": tasks}, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Find failed or missing Slurm tasks from a masking controller log.")
    parser.add_argument("--controller-log", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument(
        "--retry-manifest",
        type=Path,
        default=None,
        help="Write a JSON manifest containing tasks selected by --retry-status so they can be resubmitted.",
    )
    parser.add_argument(
        "--retry-status",
        choices=["permanent-failure", "non-success"],
        default="permanent-failure",
        help="Which tasks to include in --retry-manifest. Defaults to permanent failures only.",
    )
    parser.add_argument("--show-success", action="store_true", help="Include successful tasks in stdout and CSV output.")
    args = parser.parse_args()

    if not args.controller_log.is_file():
        raise SystemExit(f"Controller log not found: {args.controller_log}")
    if not args.manifest.is_file():
        raise SystemExit(f"Manifest not found: {args.manifest}")

    tasks = read_manifest(args.manifest)
    states = parse_controller_log(args.controller_log)
    rows = [classify(task, states.get(int(task["task_id"]))) for task in tasks]
    report_rows = rows if args.show_success else [row for row in rows if row["status"] != "success"]

    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["status"]] += 1

    print(f"manifest_tasks={len(tasks)}")
    for status in ("success", "permanent_failure", "incomplete_or_interrupted", "never_submitted"):
        print(f"{status}={counts[status]}")
    print(f"reported_rows={len(report_rows)}")

    if report_rows:
        print()
        print("task_id\tstatus\tconfig\tseed\tsplit\tresume_checkpoint\tattempts\tjob_logs")
        for row in report_rows:
            print(
                "\t".join(
                    [
                        str(row["task_id"]),
                        str(row["status"]),
                        str(row["config"]),
                        str(row["seed"]),
                        str(row["split"]),
                        str(row["resume_checkpoint"]),
                        str(row["attempts"]),
                        str(row["job_logs"]),
                    ]
                )
            )

    if args.output_csv is not None:
        write_csv(args.output_csv, report_rows)
        print()
        print(f"wrote_csv={args.output_csv}")

    if args.retry_manifest is not None:
        if args.retry_status == "permanent-failure":
            retry_rows = [row for row in rows if row["status"] == "permanent_failure"]
        else:
            retry_rows = [row for row in rows if row["status"] != "success"]
        write_retry_manifest(args.retry_manifest, retry_rows)
        print(f"wrote_retry_manifest={args.retry_manifest}")
        print(f"retry_tasks={len(retry_rows)}")


if __name__ == "__main__":
    main()
