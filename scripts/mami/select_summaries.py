import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_METRICS = ("MRAE_avg", "NAE_NDVI_avg", "NAE_NDRE_avg")
DEFAULT_WEIGHTS = (0.50, 0.25, 0.25)
DEFAULT_TIE_BREAK_METRICS = ("MRAE_median", "NAE_NDVI_median", "NAE_NDRE_median")

# Add metrics here only if larger values are better.
HIGHER_IS_BETTER_METRICS: set[str] = set()


@dataclass
class DatasetRun:
    model_name: str
    dataset_label: str
    dataset_root: Path
    directory: Path
    summary_path: Path
    metrics: dict[str, float]
    score: float = 0.0


@dataclass
class AggregatedRun:
    model_name: str
    runs_by_dataset: dict[str, DatasetRun]
    avg_metrics: dict[str, float]
    score: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select the best model directories from one or more result roots containing "
            "run subdirectories with summary.json files."
        )
    )
    parser.add_argument(
        "--results-dirs",
        nargs="+",
        type=Path,
        required=True,
        help=(
            "One or more root directories containing run subdirectories with summary.json files. "
            "Example: --results-dirs results/vi_dataset_a results/vi_dataset_b"
        ),
    )
    parser.add_argument(
        "-n",
        "--top-k",
        type=int,
        default=10,
        help="Number of top models to print.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help=(
            "Metrics used for ranking. Lower is better for error metrics such as "
            "MRAE, NAE, MSE, RMSE, and SAM. "
            f"Default: {' '.join(DEFAULT_METRICS)}"
        ),
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        help=(
            "Optional weights aligned with --metrics. "
            "Default for the default metrics is: 0.5 0.25 0.25"
        ),
    )
    parser.add_argument(
        "--tie-break-metrics",
        nargs="+",
        default=list(DEFAULT_TIE_BREAK_METRICS),
        help=(
            "Metrics used when overall scores are equal or very close. "
            f"Default: {' '.join(DEFAULT_TIE_BREAK_METRICS)}"
        ),
    )
    parser.add_argument(
        "--max-metric",
        action="append",
        default=[],
        metavar="METRIC=VALUE",
        help=(
            "Optional guardrail filter. Exclude a model if any present dataset has METRIC > VALUE. "
            "Can be given multiple times, e.g. "
            "--max-metric MRAE_RE_avg=0.15 --max-metric MRAE_NIR_avg=0.15"
        ),
    )
    parser.add_argument(
        "--allow-missing-datasets",
        action="store_true",
        help=(
            "Allow models that are not present in every dataset root. "
            "By default, incomplete models are skipped."
        ),
    )
    parser.add_argument(
        "--separator",
        default="\n",
        help="Separator used when printing selected model names with --names-only.",
    )
    parser.add_argument(
        "--names-only",
        action="store_true",
        help="Print only the selected model names.",
    )
    return parser.parse_args()


def resolve_metric_weights(
    metrics: Iterable[str],
    weights: Iterable[float] | None,
) -> dict[str, float]:
    metric_names = tuple(metrics)

    if weights is None:
        if metric_names == DEFAULT_METRICS:
            return {
                metric: weight
                for metric, weight in zip(metric_names, DEFAULT_WEIGHTS, strict=True)
            }
        return {metric: 1.0 for metric in metric_names}

    weight_values = tuple(weights)
    if len(weight_values) != len(metric_names):
        raise ValueError(
            "The number of --weights values must match the number of --metrics values."
        )

    if any(weight < 0 for weight in weight_values):
        raise ValueError("Metric weights must be non-negative.")

    total_weight = sum(weight_values)
    if total_weight == 0:
        raise ValueError("At least one metric weight must be greater than zero.")

    return {
        metric: weight
        for metric, weight in zip(metric_names, weight_values, strict=True)
    }


def parse_metric_thresholds(entries: Iterable[str]) -> dict[str, float]:
    thresholds: dict[str, float] = {}

    for entry in entries:
        if "=" not in entry:
            raise ValueError(
                f"Invalid --max-metric value '{entry}'. Expected format METRIC=VALUE."
            )

        metric, raw_value = entry.split("=", 1)
        metric = metric.strip()
        raw_value = raw_value.strip()

        if not metric:
            raise ValueError(f"Invalid --max-metric value '{entry}': missing metric name.")

        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid --max-metric value '{entry}': '{raw_value}' is not a float."
            ) from exc

        thresholds[metric] = value

    return thresholds


def is_higher_better(metric: str) -> bool:
    if metric in HIGHER_IS_BETTER_METRICS:
        return True

    if metric.startswith("PSNR"):
        return True
    if metric.startswith("SSIM"):
        return True

    return False


def make_dataset_labels(results_dirs: list[Path]) -> dict[Path, str]:
    counts = Counter(path.name for path in results_dirs)
    labels: dict[Path, str] = {}

    for path in results_dirs:
        if path.name and counts[path.name] == 1:
            labels[path] = path.name
        else:
            labels[path] = str(path)

    return labels


def load_runs_for_dataset(
    results_dir: Path,
    dataset_label: str,
    required_metrics: Iterable[str],
) -> dict[str, DatasetRun]:
    runs: dict[str, DatasetRun] = {}
    metric_names = tuple(dict.fromkeys(required_metrics))

    for summary_path in sorted(results_dir.glob("*/summary.json")):
        with summary_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        missing_metrics = [metric for metric in metric_names if metric not in data]
        if missing_metrics:
            missing = ", ".join(missing_metrics)
            raise KeyError(f"{summary_path} is missing required metrics: {missing}")

        metric_values: dict[str, float] = {}
        for metric in metric_names:
            value = float(data[metric])
            if not math.isfinite(value):
                raise ValueError(f"{summary_path} has non-finite value for metric '{metric}': {value}")
            metric_values[metric] = value

        model_name = summary_path.parent.name
        if model_name in runs:
            raise ValueError(
                f"Duplicate model directory name '{model_name}' found under {results_dir}."
            )

        runs[model_name] = DatasetRun(
            model_name=model_name,
            dataset_label=dataset_label,
            dataset_root=results_dir,
            directory=summary_path.parent,
            summary_path=summary_path,
            metrics=metric_values,
        )

    if not runs:
        raise FileNotFoundError(f"No summary.json files found under {results_dir}")

    return runs


def score_dataset_runs(
    runs: dict[str, DatasetRun],
    metrics: Iterable[str],
    metric_weights: dict[str, float],
) -> None:
    metric_names = tuple(metrics)
    total_weight = sum(metric_weights[metric] for metric in metric_names)
    run_list = list(runs.values())

    for run in run_list:
        run.score = 0.0

    for metric in metric_names:
        higher_is_better = is_higher_better(metric)
        weight = metric_weights[metric]
        values = [run.metrics[metric] for run in run_list]
        min_value = min(values)
        max_value = max(values)
        span = max_value - min_value

        for run in run_list:
            if span == 0:
                normalized = 0.0
            else:
                normalized = (run.metrics[metric] - min_value) / span
                if higher_is_better:
                    normalized = 1.0 - normalized
            run.score += normalized * weight

    for run in run_list:
        run.score /= total_weight


def apply_guardrails_to_model(
    runs_by_dataset: dict[str, DatasetRun],
    thresholds: dict[str, float],
) -> bool:
    if not thresholds:
        return True

    for run in runs_by_dataset.values():
        for metric, max_value in thresholds.items():
            if run.metrics[metric] > max_value:
                return False
    return True


def aggregate_models(
    runs_by_root: dict[str, dict[str, DatasetRun]],
    ranking_metrics: Iterable[str],
    tie_break_metrics: Iterable[str],
    thresholds: dict[str, float],
    allow_missing_datasets: bool,
) -> tuple[list[AggregatedRun], dict[str, list[str]]]:
    dataset_labels = tuple(runs_by_root.keys())
    all_model_names = sorted({model for runs in runs_by_root.values() for model in runs})
    missing_by_model: dict[str, list[str]] = {}
    aggregated_runs: list[AggregatedRun] = []

    metric_names = tuple(dict.fromkeys([*ranking_metrics, *tie_break_metrics]))

    for model_name in all_model_names:
        present_runs = {
            dataset_label: runs[model_name]
            for dataset_label, runs in runs_by_root.items()
            if model_name in runs
        }

        missing = [dataset_label for dataset_label in dataset_labels if dataset_label not in present_runs]
        if missing:
            missing_by_model[model_name] = missing
            if not allow_missing_datasets:
                continue

        if not apply_guardrails_to_model(present_runs, thresholds):
            continue

        avg_metrics = {
            metric: sum(run.metrics[metric] for run in present_runs.values()) / len(present_runs)
            for metric in metric_names
        }

        score = sum(run.score for run in present_runs.values()) / len(present_runs)

        aggregated_runs.append(
            AggregatedRun(
                model_name=model_name,
                runs_by_dataset=present_runs,
                avg_metrics=avg_metrics,
                score=score,
            )
        )

    if not aggregated_runs:
        raise ValueError("No models remain after aggregation. Check missing models or guardrails.")

    return aggregated_runs, missing_by_model


def metric_sort_value(value: float, metric: str) -> float:
    return -value if is_higher_better(metric) else value


def sort_aggregated_runs(
    runs: list[AggregatedRun],
    ranking_metrics: Iterable[str],
    tie_break_metrics: Iterable[str],
) -> list[AggregatedRun]:
    ranking_metric_names = tuple(ranking_metrics)
    tie_metric_names = tuple(tie_break_metrics)

    return sorted(
        runs,
        key=lambda run: (
            run.score,
            *(
                metric_sort_value(run.avg_metrics[metric], metric)
                for metric in tie_metric_names
            ),
            *(
                metric_sort_value(run.avg_metrics[metric], metric)
                for metric in ranking_metric_names
            ),
            run.model_name,
        ),
    )


def print_missing_models_warning(
    missing_by_model: dict[str, list[str]],
    allow_missing_datasets: bool,
) -> None:
    if not missing_by_model:
        return

    action = "including incomplete models" if allow_missing_datasets else "skipping incomplete models"
    print(f"Warning: found models missing from one or more datasets; {action}.", file=sys.stderr)

    preview_limit = 10
    for index, (model_name, missing) in enumerate(sorted(missing_by_model.items()), start=1):
        if index > preview_limit:
            remaining = len(missing_by_model) - preview_limit
            print(f"... and {remaining} more incomplete model(s).", file=sys.stderr)
            break
        print(
            f"  {model_name}: missing from {', '.join(missing)}",
            file=sys.stderr,
        )


def print_ranked_runs(
    runs: list[AggregatedRun],
    ranking_metrics: Iterable[str],
    tie_break_metrics: Iterable[str],
    dataset_labels: list[str],
    names_only: bool,
    separator: str,
) -> None:
    if names_only:
        print(separator.join(run.model_name for run in runs))
        return

    ranking_metric_names = tuple(ranking_metrics)
    tie_metric_names = tuple(tie_break_metrics)

    header = [
        "rank",
        "model",
        "score",
        "datasets",
        *ranking_metric_names,
        *tie_metric_names,
        *(f"{label}_score" for label in dataset_labels),
    ]
    rows = [header]

    for index, run in enumerate(runs, start=1):
        rows.append(
            [
                str(index),
                run.model_name,
                f"{run.score:.6f}",
                f"{len(run.runs_by_dataset)}/{len(dataset_labels)}",
                *(f"{run.avg_metrics[metric]:.6f}" for metric in ranking_metric_names),
                *(f"{run.avg_metrics[metric]:.6f}" for metric in tie_metric_names),
                *(
                    f"{run.runs_by_dataset[label].score:.6f}" if label in run.runs_by_dataset else "-"
                    for label in dataset_labels
                ),
            ]
        )

    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for row in rows:
        print("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


def main() -> None:
    args = parse_args()

    metric_weights = resolve_metric_weights(args.metrics, args.weights)
    guardrails = parse_metric_thresholds(args.max_metric)

    required_metrics = tuple(
        dict.fromkeys(
            [*args.metrics, *args.tie_break_metrics, *guardrails.keys()]
        )
    )

    dataset_labels_by_path = make_dataset_labels(args.results_dirs)
    dataset_labels = [dataset_labels_by_path[path] for path in args.results_dirs]

    runs_by_root: dict[str, dict[str, DatasetRun]] = {}
    for results_dir in args.results_dirs:
        dataset_label = dataset_labels_by_path[results_dir]
        dataset_runs = load_runs_for_dataset(results_dir, dataset_label, required_metrics)
        score_dataset_runs(dataset_runs, args.metrics, metric_weights)
        runs_by_root[dataset_label] = dataset_runs

    aggregated_runs, missing_by_model = aggregate_models(
        runs_by_root=runs_by_root,
        ranking_metrics=args.metrics,
        tie_break_metrics=args.tie_break_metrics,
        thresholds=guardrails,
        allow_missing_datasets=args.allow_missing_datasets,
    )

    print_missing_models_warning(missing_by_model, args.allow_missing_datasets)

    ranked_runs = sort_aggregated_runs(
        aggregated_runs,
        ranking_metrics=args.metrics,
        tie_break_metrics=args.tie_break_metrics,
    )

    selected_runs = ranked_runs[: min(args.top_k, len(ranked_runs))]
    print_ranked_runs(
        selected_runs,
        ranking_metrics=args.metrics,
        tie_break_metrics=args.tie_break_metrics,
        dataset_labels=dataset_labels,
        names_only=args.names_only,
        separator=args.separator,
    )


if __name__ == "__main__":
    main()