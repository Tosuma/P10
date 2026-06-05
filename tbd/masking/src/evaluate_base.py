from __future__ import annotations

import argparse
from pathlib import Path

from src.config import dump_config, load_config
from src.evaluate import run_split_evaluation
from src.model import build_model, resolve_model_config
from src.train import prepare_normalization
from src.utils import (
    device_from_config,
    ensure_dir,
    get_git_commit,
    package_versions,
    resolve_seed,
    set_seed,
    setup_file_logger,
    timestamp_utc,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an unfine-tuned base segmentation model from config.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    config["seed"] = resolve_seed(config.get("seed"))
    set_seed(config["seed"])
    config["model"] = resolve_model_config(config.get("model"))

    if args.output_dir:
        run_dir = ensure_dir(Path(args.output_dir))
    else:
        run_name = f"{config['experiment_name']}_baseline_seed{config['seed']}_{timestamp_utc()}"
        run_dir = ensure_dir(Path(config["paths"]["run_root"]) / run_name)
    evaluation_dir = ensure_dir(run_dir / "evaluation" / args.split)
    log_dir = ensure_dir(run_dir / "logs")
    logger = setup_file_logger("BaseEvaluator", log_dir / "execution.log")

    dump_config(run_dir / "config.yaml", config)
    write_json(
        run_dir / "run_metadata.json",
        {
            "git_commit": get_git_commit(),
            "seed": config["seed"],
            "packages": package_versions(["torch", "numpy", "PIL"]),
            "run_kind": "baseline",
            "evaluation_split": args.split,
        },
    )
    logger.info("Baseline evaluation initialized at %s", run_dir.resolve())
    logger.info("Loaded config from %s", config.get("_config_path"))
    logger.info(
        "Using model architecture=%s encoder=%s encoder_weights=%s",
        config["model"]["architecture"],
        config["model"]["encoder_name"],
        config["model"]["encoder_weights"],
    )

    normalization = prepare_normalization(config, run_dir)
    device = device_from_config(config.get("device"))
    logger.info("Resolved runtime device to %s", device)
    logger.info("Prepared normalization from source=%s", normalization.get("source", "unknown"))

    model = build_model(int(config["in_channels"]), config["model"]).to(device)
    model.eval()
    run_split_evaluation(model, config, args.split, evaluation_dir, device, logger)
    overall_metrics_path = evaluation_dir / "overall_metrics.json"
    if not overall_metrics_path.is_file():
        raise RuntimeError(f"Baseline evaluation did not create expected metrics file: {overall_metrics_path}")
    logger.info("Baseline overall metrics written to %s", overall_metrics_path)
    logger.info("Baseline evaluation completed")
    print(str(run_dir.resolve()))


if __name__ == "__main__":
    main()
