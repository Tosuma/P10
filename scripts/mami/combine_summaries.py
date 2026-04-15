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

# Add metrics here only if larger values are better.
HIGHER_IS_BETTER_METRICS: set[str] = set()


@dataclass
class DatasetSummary:
    model_name: str
    dataset_label: str
    summary_path: Path
    metrics: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine model summary.json files from multiple dataset result roots. "
            "Each combined metric is the arithmetic mean of the matching metric "
            "from the available dataset summaries."
        )
    )
    parser.add_argument(
        "--results-dirs",
        nargs="+",
        type=Path,
        required=True,
        help=(
            "Two or more root directories containing model subdirectories with "
            "summary.json files. Example: --results-dirs results/vi/stage1---sri-lanka "
            "results/vi/stage1---weedy-rice"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination root for combined model summary.json files.",
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
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help=(
            "Metrics used for the optional weighted combined score. Lower is better "
            "for error metrics such as MRAE, NAE, MSE, RMSE, and SAM. "
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
        "--score-key",
        default="combined_score",
        help=(
            "Metric key to write for the weighted score. Use --no-score to disable. "
            "Default: combined_score"
        ),
    )
    parser.add_argument(
        "--no-score",
        action="store_true",
        help="Do not add a weighted score field to each combined summary.json.",
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


def load_summary_metrics(summary_path: Path) -> dict[str, float]:
    with summary_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise TypeError(f"{summary_path} must contain a JSON object.")

    metrics: dict[str, float] = {}
    for metric, raw_value in data.items():
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{summary_path} has non-numeric value for metric '{metric}': {raw_value!r}"
            ) from exc

        if math.isinf(value):
            raise ValueError(
                f"{summary_path} has infinite value for metric '{metric}': {value}"
            )

        metrics[metric] = value

    return metrics


def load_summaries_for_dataset(
    results_dir: Path,
    dataset_label: str,
) -> dict[str, DatasetSummary]:
    if not results_dir.exists():
        raise FileNotFoundError(f"Result directory does not exist: {results_dir}")
    if not results_dir.is_dir():
        raise NotADirectoryError(f"Result path is not a directory: {results_dir}")

    summaries: dict[str, DatasetSummary] = {}

    for summary_path in sorted(results_dir.glob("*/summary.json")):
        model_name = summary_path.parent.name
        if model_name in summaries:
            raise ValueError(
                f"Duplicate model directory name '{model_name}' found under {results_dir}."
            )

        summaries[model_name] = DatasetSummary(
            model_name=model_name,
            dataset_label=dataset_label,
            summary_path=summary_path,
            metrics=load_summary_metrics(summary_path),
        )

    if not summaries:
        raise FileNotFoundError(f"No summary.json files found under {results_dir}")

    return summaries


def collect_missing_models(
    summaries_by_dataset: dict[str, dict[str, DatasetSummary]],
) -> dict[str, list[str]]:
    dataset_labels = tuple(summaries_by_dataset.keys())
    all_model_names = sorted(
        {model_name for summaries in summaries_by_dataset.values() for model_name in summaries}
    )
    missing_by_model: dict[str, list[str]] = {}

    for model_name in all_model_names:
        missing = [
            dataset_label
            for dataset_label in dataset_labels
            if model_name not in summaries_by_dataset[dataset_label]
        ]
        if missing:
            missing_by_model[model_name] = missing

    return missing_by_model


def print_missing_models_warning(
    missing_by_model: dict[str, list[str]],
    allow_missing_datasets: bool,
) -> None:
    if not missing_by_model:
        return

    action = "including incomplete models" if allow_missing_datasets else "skipping incomplete models"
    print(
        f"Warning: found models missing from one or more datasets; {action}.",
        file=sys.stderr,
    )

    preview_limit = 10
    for index, (model_name, missing) in enumerate(sorted(missing_by_model.items()), start=1):
        if index > preview_limit:
            remaining = len(missing_by_model) - preview_limit
            print(f"... and {remaining} more incomplete model(s).", file=sys.stderr)
            break
        print(f"  {model_name}: missing from {', '.join(missing)}", file=sys.stderr)


def combine_model_summaries(
    model_name: str,
    present_summaries: list[DatasetSummary],
) -> dict[str, float]:
    if not present_summaries:
        raise ValueError(f"No summaries available for model '{model_name}'.")

    expected_metrics = tuple(present_summaries[0].metrics.keys())
    expected_set = set(expected_metrics)

    for summary in present_summaries[1:]:
        actual_set = set(summary.metrics.keys())
        if actual_set != expected_set:
            missing = sorted(expected_set - actual_set)
            extra = sorted(actual_set - expected_set)
            details: list[str] = []
            if missing:
                details.append(f"missing: {', '.join(missing)}")
            if extra:
                details.append(f"extra: {', '.join(extra)}")
            raise ValueError(
                f"{summary.summary_path} has mismatched metric keys for model "
                f"'{model_name}' ({'; '.join(details)})."
            )

    return {
        metric: sum(summary.metrics[metric] for summary in present_summaries)
        / len(present_summaries)
        for metric in expected_metrics
    }


def write_combined_summary(
    output_dir: Path,
    model_name: str,
    combined_metrics: dict[str, float],
) -> None:
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    summary_path = model_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(combined_metrics, handle, indent=4)
        handle.write("\n")


def add_weighted_scores(
    combined_by_model: dict[str, dict[str, float]],
    metrics: Iterable[str],
    metric_weights: dict[str, float],
    score_key: str,
) -> None:
    metric_names = tuple(metrics)
    total_weight = sum(metric_weights[metric] for metric in metric_names)

    for metric in metric_names:
        missing_models = [
            model_name
            for model_name, combined_metrics in combined_by_model.items()
            if metric not in combined_metrics
        ]
        if missing_models:
            preview = ", ".join(missing_models[:10])
            raise KeyError(
                f"Metric '{metric}' is missing from combined summaries for: {preview}"
            )

        values = []
        for model_name, combined_metrics in combined_by_model.items():
            value = combined_metrics[metric]
            if not math.isfinite(value):
                raise ValueError(
                    f"Metric '{metric}' for model '{model_name}' must be finite "
                    "to compute weighted scores."
                )
            values.append(value)

        higher_is_better = is_higher_better(metric)
        weight = metric_weights[metric]
        min_value = min(values)
        max_value = max(values)
        span = max_value - min_value

        for combined_metrics in combined_by_model.values():
            if score_key not in combined_metrics:
                combined_metrics[score_key] = 0.0

            if span == 0:
                normalized = 0.0
            else:
                normalized = (combined_metrics[metric] - min_value) / span
                if higher_is_better:
                    normalized = 1.0 - normalized

            combined_metrics[score_key] += normalized * weight

    for combined_metrics in combined_by_model.values():
        combined_metrics[score_key] /= total_weight


def main() -> None:
    args = parse_args()

    if len(args.results_dirs) < 2:
        raise ValueError("--results-dirs must include at least two directories.")
    if not args.no_score and not args.score_key:
        raise ValueError("--score-key must not be empty unless --no-score is used.")

    metric_weights = resolve_metric_weights(args.metrics, args.weights)

    dataset_labels_by_path = make_dataset_labels(args.results_dirs)

    summaries_by_dataset: dict[str, dict[str, DatasetSummary]] = {}
    for results_dir in args.results_dirs:
        dataset_label = dataset_labels_by_path[results_dir]
        summaries_by_dataset[dataset_label] = load_summaries_for_dataset(
            results_dir,
            dataset_label,
        )

    missing_by_model = collect_missing_models(summaries_by_dataset)
    print_missing_models_warning(missing_by_model, args.allow_missing_datasets)

    dataset_labels = tuple(summaries_by_dataset.keys())
    all_model_names = sorted(
        {model_name for summaries in summaries_by_dataset.values() for model_name in summaries}
    )

    combined_by_model: dict[str, dict[str, float]] = {}
    for model_name in all_model_names:
        present_summaries = [
            summaries_by_dataset[dataset_label][model_name]
            for dataset_label in dataset_labels
            if model_name in summaries_by_dataset[dataset_label]
        ]

        if len(present_summaries) != len(dataset_labels) and not args.allow_missing_datasets:
            continue

        combined_metrics = combine_model_summaries(model_name, present_summaries)
        combined_by_model[model_name] = combined_metrics

    if not combined_by_model:
        raise ValueError("No combined summaries were written.")

    if not args.no_score:
        add_weighted_scores(
            combined_by_model=combined_by_model,
            metrics=args.metrics,
            metric_weights=metric_weights,
            score_key=args.score_key,
        )

    for model_name, combined_metrics in combined_by_model.items():
        write_combined_summary(args.output_dir, model_name, combined_metrics)

    print(f"Wrote {len(combined_by_model)} combined summary.json file(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
