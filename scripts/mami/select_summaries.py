import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_METRICS = ("MRAE_avg", "NAE_NDVI_avg", "NAE_NDRE_avg")
DEFAULT_WEIGHTS = (0.50, 0.25, 0.25)
DEFAULT_TIE_BREAK_METRICS = ("MRAE_median", "NAE_NDVI_median", "NAE_NDRE_median")

HIGHER_IS_BETTER_METRICS: set[str] = set()


@dataclass
class RunSummary:
    name: str
    directory: Path
    summary_path: Path
    metrics: dict[str, float]
    score: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select the best run directories from results/vi based on summary.json metrics."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/vi"),
        help="Directory containing run subdirectories with summary.json files.",
    )
    parser.add_argument(
        "-n",
        "--top-k",
        type=int,
        default=10,
        help="Number of top runs to print.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help=(
            "Metrics to use for ranking. Lower is better for error metrics such as "
            "MRAE, NAE, MSE, RMSE, and SAM. "
            f"Default: {' '.join(DEFAULT_METRICS)}"
        ),
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        help=(
            "Optional weights aligned with --metrics. Higher values make a metric "
            "more important when computing the final score. "
            "For Stage 1, a good default is: "
            "--metrics MRAE_avg NAE_NDVI_avg NAE_NDRE_avg "
            "--weights 0.5 0.25 0.25"
        ),
    )
    parser.add_argument(
        "--tie-break-metrics",
        nargs="+",
        default=list(DEFAULT_TIE_BREAK_METRICS),
        help=(
            "Metrics used for sorting when composite scores are equal or very close. "
            f"Default: {' '.join(DEFAULT_TIE_BREAK_METRICS)}"
        ),
    )
    parser.add_argument(
        "--max-metric",
        action="append",
        default=[],
        metavar="METRIC=VALUE",
        help=(
            "Optional guardrail filter. Exclude runs where METRIC > VALUE. "
            "Can be given multiple times, e.g. "
            "--max-metric MRAE_RE_avg=0.15 --max-metric MRAE_NIR_avg=0.15"
        ),
    )
    parser.add_argument(
        "--separator",
        default="\n",
        help="Separator used when printing the selected directory names.",
    )
    parser.add_argument(
        "--names-only",
        action="store_true",
        help="Print only the selected directory names.",
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

    # Sensible generic handling if you later rank on these metrics.
    if metric.startswith("PSNR"):
        return True
    if metric.startswith("SSIM"):
        return True

    return False


def load_runs(results_dir: Path, required_metrics: Iterable[str]) -> list[RunSummary]:
    runs: list[RunSummary] = []
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

        runs.append(
            RunSummary(
                name=summary_path.parent.name,
                directory=summary_path.parent,
                summary_path=summary_path,
                metrics=metric_values,
            )
        )

    if not runs:
        raise FileNotFoundError(f"No summary.json files found under {results_dir}")

    return runs


def apply_guardrails(
    runs: list[RunSummary],
    thresholds: dict[str, float],
) -> list[RunSummary]:
    if not thresholds:
        return runs

    filtered_runs: list[RunSummary] = []
    for run in runs:
        keep = True
        for metric, max_value in thresholds.items():
            if run.metrics[metric] > max_value:
                keep = False
                break
        if keep:
            filtered_runs.append(run)

    if not filtered_runs:
        formatted = ", ".join(f"{metric}<={value}" for metric, value in thresholds.items())
        raise ValueError(f"No runs remain after applying guardrails: {formatted}")

    return filtered_runs


def score_runs(
    runs: list[RunSummary],
    metrics: Iterable[str],
    metric_weights: dict[str, float],
) -> None:
    metric_names = tuple(metrics)
    total_weight = sum(metric_weights[metric] for metric in metric_names)

    for run in runs:
        run.score = 0.0

    for metric in metric_names:
        higher_is_better = is_higher_better(metric)
        weight = metric_weights[metric]
        values = [run.metrics[metric] for run in runs]
        min_value = min(values)
        max_value = max(values)
        span = max_value - min_value

        for run in runs:
            if span == 0:
                normalized = 0.0
            else:
                normalized = (run.metrics[metric] - min_value) / span
                if higher_is_better:
                    normalized = 1.0 - normalized
            run.score += normalized * weight

    for run in runs:
        run.score /= total_weight


def metric_sort_value(value: float, metric: str) -> float:
    return -value if is_higher_better(metric) else value


def sort_runs(
    runs: list[RunSummary],
    ranking_metrics: Iterable[str],
    tie_break_metrics: Iterable[str],
) -> list[RunSummary]:
    rank_metric_names = tuple(ranking_metrics)
    tie_metric_names = tuple(tie_break_metrics)

    return sorted(
        runs,
        key=lambda run: (
            run.score,
            *(
                metric_sort_value(run.metrics[metric], metric)
                for metric in tie_metric_names
            ),
            *(
                metric_sort_value(run.metrics[metric], metric)
                for metric in rank_metric_names
            ),
            run.name,
        ),
    )


def print_ranked_runs(
    runs: list[RunSummary],
    ranking_metrics: Iterable[str],
    tie_break_metrics: Iterable[str],
    names_only: bool,
    separator: str,
) -> None:
    if names_only:
        print(separator.join(run.name for run in runs))
        return

    ranking_metric_names = tuple(ranking_metrics)
    tie_metric_names = tuple(tie_break_metrics)

    header = ["rank", "directory", "score", *ranking_metric_names, *tie_metric_names]
    rows = [header]

    for index, run in enumerate(runs, start=1):
        rows.append(
            [
                str(index),
                run.name,
                f"{run.score:.6f}",
                *(f"{run.metrics[metric]:.6f}" for metric in ranking_metric_names),
                *(f"{run.metrics[metric]:.6f}" for metric in tie_metric_names),
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

    runs = load_runs(args.results_dir, required_metrics)
    runs = apply_guardrails(runs, guardrails)
    score_runs(runs, args.metrics, metric_weights)
    ranked_runs = sort_runs(runs, args.metrics, args.tie_break_metrics)

    selected_runs = ranked_runs[: min(args.top_k, len(ranked_runs))]
    print_ranked_runs(
        selected_runs,
        args.metrics,
        args.tie_break_metrics,
        args.names_only,
        args.separator,
    )


if __name__ == "__main__":
    main()