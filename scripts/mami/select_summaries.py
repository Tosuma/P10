import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_METRICS = ("MRAE_avg", "NAE_NDVI_avg", "NAE_NDRE_avg")
HIGHER_IS_BETTER_METRICS = {"NAE_NDVI_avg", "NAE_NDRE_avg"}


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
            "Metrics to use for ranking. Lower is better for MRAE-style metrics, "
            "while higher is better for NAE_NDVI_avg and NAE_NDRE_avg. "
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
            "Example: --metrics MRAE_avg NAE_NDVI_avg NAE_NDRE_avg "
            "--weights 1.0 0.5 0.5"
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
        help="Print only the selected directory names on one line.",
    )
    return parser.parse_args()


def resolve_metric_weights(metrics: Iterable[str], weights: Iterable[float] | None) -> dict[str, float]:
    metric_names = tuple(metrics)

    if weights is None:
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


def load_runs(results_dir: Path, metrics: Iterable[str]) -> list[RunSummary]:
    runs: list[RunSummary] = []
    required_metrics = tuple(metrics)

    for summary_path in sorted(results_dir.glob("*/summary.json")):
        with summary_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        missing_metrics = [metric for metric in required_metrics if metric not in data]
        if missing_metrics:
            missing = ", ".join(missing_metrics)
            raise KeyError(f"{summary_path} is missing required metrics: {missing}")

        metric_values = {metric: float(data[metric]) for metric in required_metrics}
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


def score_runs(
    runs: list[RunSummary],
    metrics: Iterable[str],
    metric_weights: dict[str, float],
) -> None:
    metric_names = tuple(metrics)
    total_weight = sum(metric_weights[metric] for metric in metric_names)

    for metric in metric_names:
        higher_is_better = metric in HIGHER_IS_BETTER_METRICS
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


def sort_runs(runs: list[RunSummary], metrics: Iterable[str]) -> list[RunSummary]:
    metric_names = tuple(metrics)
    return sorted(
        runs,
        key=lambda run: (
            run.score,
            *(
                -run.metrics[metric]
                if metric in HIGHER_IS_BETTER_METRICS
                else run.metrics[metric]
                for metric in metric_names
            ),
            run.name,
        ),
    )


def print_ranked_runs(runs: list[RunSummary], metrics: Iterable[str], names_only: bool, separator: str) -> None:
    if names_only:
        print(separator.join(run.name for run in runs))
        return

    header = ["rank", "directory", "score", *metrics]
    rows = [header]
    for index, run in enumerate(runs, start=1):
        rows.append(
            [
                str(index),
                run.name,
                f"{run.score:.6f}",
                *(f"{run.metrics[metric]:.6f}" for metric in metrics),
            ]
        )

    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for row in rows:
        print("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)))

    if names_only:
        print()
        print("Selected directory names:")
        print(separator.join(run.name for run in runs))


def main() -> None:
    args = parse_args()
    metric_weights = resolve_metric_weights(args.metrics, args.weights)
    runs = load_runs(args.results_dir, args.metrics)
    score_runs(runs, args.metrics, metric_weights)
    ranked_runs = sort_runs(runs, args.metrics)
    selected_runs = ranked_runs[: min(args.top_k, len(ranked_runs))]
    print_ranked_runs(selected_runs, args.metrics, args.names_only, args.separator)


if __name__ == "__main__":
    main()
