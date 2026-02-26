# SPDX-License-Identifier: MIT
# Copyright (c) 2025 <Hugin J. Zachariasen, Magnus H. Jensen, Tobias S. Madsen>.

"""Distributed inference (Slurm/torchrun friendly).

Usage (single node):
  python inference.py --data_type Kazakhstan --model path/to/model.pth

Usage (multi-GPU, sharded dataset):
  torchrun --standalone --nproc_per_node=4 inference.py --device cuda --data_type Kazakhstan --model path/to/model.pth

Notes:
- This is *data parallel* inference: each rank processes a disjoint subset of images.
- We don't *need* to wrap the model in DDP for inference; DistributedSampler is
  enough to avoid duplicated work.
"""

import os
import logging
from pathlib import Path

import argparse
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

import matplotlib.pyplot as plt
import cv2

from mstpp.mstpp import MST_Plus_Plus
from data_carrier import load_east_kaz, load_sri_lanka, load_weedy_rice, DataCarrier

# Reduce noisy OpenCV logging if present
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

# --- DDP environment (torchrun / Slurm) ---
RANK = int(os.environ.get("RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))


def _setup_logging() -> logging.Logger:
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / (f"inference_rank{RANK}.log" if WORLD_SIZE > 1 else "inference.log")
    logging.basicConfig(
        level=(logging.INFO if RANK == 0 else logging.WARNING),
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="w"),
        ],
    )
    return logging.getLogger(__name__)


logger = _setup_logging()


def _select_device(device_str: str | None) -> torch.device:
    """Return a torch.device, respecting torchrun local rank for CUDA."""
    if device_str == None:
        use_cuda = torch.cuda.is_available()
    elif device_str == "cuda":
        use_cuda = torch.cuda.is_available()
        if not use_cuda and RANK == 0:
            logger.warning("--device cuda requested, but CUDA is not available. Falling back to CPU.")
    else:
        use_cuda = False

    if use_cuda:
        torch.cuda.set_device(LOCAL_RANK)
        return torch.device("cuda", LOCAL_RANK)

    return torch.device("cpu")


def _init_distributed(device: torch.device) -> bool:
    """Initialize torch.distributed if launched with torchrun (WORLD_SIZE>1)."""
    if WORLD_SIZE <= 1:
        return False

    if dist.is_initialized():
        return True

    backend = "nccl" if device.type == "cuda" else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    dist.barrier()

    if RANK == 0:
        logger.info(f"[DDP] Initialized process group: backend={backend} world_size={WORLD_SIZE}")

    return True


def _cleanup_distributed(enabled: bool):
    if enabled and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _load_state_dict_compat(model: torch.nn.Module, checkpoint_path: str, device: torch.device):
    """Load a checkpoint that may or may not have 'module.' prefixes / wrapper dicts."""
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    model_sd = model.state_dict()
    filtered = {}

    for k, v in state_dict.items():
        key = k
        if key.startswith("module."):
            key = key[len("module.") :]

        if key in model_sd and hasattr(v, "size") and v.size() == model_sd[key].size():
            filtered[key] = v

    missing = set(model_sd.keys()) - set(filtered.keys())
    extra = set(state_dict.keys()) - {k if not k.startswith("module.") else k[len("module.") :] for k in filtered.keys()}

    if RANK == 0:
        logger.info(
            f"Loading checkpoint: kept {len(filtered)} params, missing {len(missing)} params, skipped {len(extra)} params"
        )

    model.load_state_dict(filtered, strict=False)


def _get_loader_function(data_type: str):
    match data_type:
        case "Sri-Lanka":
            return load_sri_lanka
        case "Kazakhstan":
            return load_east_kaz
        case "Weedy-Rice":
            return load_weedy_rice
        case _:
            raise ValueError(f"Unknown dataset type: {data_type}")


def run(
    root_dir: str = "data/",
    data_type: str = "Sri-Lanka",
    save_dir: str = "results",
    single: bool = False,
    single_picture: str | None = None,
    amount: str = "Full",
    model_path: str = "model_final.pkl",
    save_images: bool = False,
    device_str: str | None = None,
):
    device = _select_device(device_str)
    ddp = _init_distributed(device)
    is_main = (RANK == 0)

    if is_main:
        logger.info(f"[Device] {device} | ddp={ddp} | world_size={WORLD_SIZE}")

    if single and ddp and not is_main:
        # Avoid duplicated work / output collisions.
        logger.warning("[DDP] --single requested; running only on rank 0.")
        _cleanup_distributed(ddp)
        return

    model = MST_Plus_Plus(in_channels=3, out_channels=4, n_feat=4, stage=3).to(device)
    _load_state_dict_compat(model, model_path, device)
    model.eval()

    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Dataset ---
    loader_fn = _get_loader_function(data_type)

    if single:
        if is_main:
            logger.info("Running single image")
        dataset = DataCarrier(Path(root_dir + (single_picture or "")), loader_fn, resize=False)
    else:
        dataset = DataCarrier(Path(root_dir), loader_fn, resize=False)

    # --- Respect "amount" globally (important under DDP) ---
    if (not single) and (amount != "Full"):
        limit = int(amount)
        limit = max(0, min(limit, len(dataset)))
        dataset = Subset(dataset, list(range(limit)))

    sampler = (
        DistributedSampler(dataset, num_replicas=WORLD_SIZE, rank=RANK, shuffle=False, drop_last=False)
        if (ddp and not single)
        else None
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    local_index = 0

    # Estimate per-rank progress only (rank 0 prints)
    if is_main:
        logger.info(f"Dataset size={len(dataset)} | per-rank batches≈{len(dataloader)}")

    for batch_idx, sample in enumerate(dataloader):
        rgb = sample["rgb"]
        target = sample["ms"]
        file_path = Path(sample["path"][0])

        if is_main:
            step = max(1, len(dataloader) // 10)
            if (batch_idx % step) == 0:
                logger.info(f"Processing rank0 [{batch_idx + 1}/{len(dataloader)}]")

        rgb_vis = rgb.permute(0, 2, 3, 1).cpu().numpy().squeeze(0)
        target_np = target.squeeze(0).cpu().numpy() if target.dim() == 4 else target.cpu().numpy()

        rgb = rgb.to(device, non_blocking=True)

        with torch.no_grad():
            output = model(rgb)
            if isinstance(output, list):
                output = output[-1]
            pred = output.squeeze(0).detach().cpu().numpy()

        pred = np.clip(pred, 0, 1)

        # Save numpy file (keeps original naming behavior)
        np.save(output_dir / file_path.stem, pred.astype(np.float32))

        if save_images:
            # Ensure filenames don't collide between ranks
            grid_name = f"validation_result_{local_index}_rank{RANK}.png"

            _, axes = plt.subplots(2, 5, figsize=(14, 5))
            axes[0, 0].imshow(rgb_vis)
            axes[0, 0].set_title("RGB Input")
            axes[0, 0].axis("off")

            for i in range(4):
                axes[0, i + 1].imshow(target_np[i], cmap="gray")
                axes[0, i + 1].set_title(f"GT Band {i + 1}")
                axes[0, i + 1].axis("off")

            for i in range(4):
                axes[1, i].imshow(pred[i], cmap="gray")
                axes[1, i].set_title(f"Pred Band {i + 1}")
                axes[1, i].axis("off")

            axes[1, 4].axis("off")
            plt.tight_layout()

            latest_name = "validation_result.png" if WORLD_SIZE == 1 else f"validation_result_rank{RANK}.png"
            plt.savefig(latest_name, dpi=150, bbox_inches="tight")
            plt.savefig(output_dir / grid_name, dpi=150, bbox_inches="tight")
            plt.close()

            for i in range(4):
                img = (pred[i] * 255).clip(0, 255).astype(np.uint8)
                band_name = f"validation_result_{local_index}_{i + 1}_rank{RANK}.JPG"
                cv2.imwrite(str(output_dir / band_name), img)

        local_index += 1

    if ddp:
        dist.barrier()
        if is_main:
            logger.info("[DDP] Inference complete on all ranks.")

    _cleanup_distributed(ddp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs inference on images.")
    parser.add_argument("--data_path", help="Path to directory with data, default=data/", default="data/")
    parser.add_argument("--single", help="One or many pictures, default=many", action=argparse.BooleanOptionalAction)
    parser.add_argument("--jpg", help="Path to single picture (used only if --single).", default=None)
    parser.add_argument(
        "--amount",
        help="Amount of pictures the eval should run through (only if --single is false). Default=Full (entire dataset)",
        default="Full",
    )
    parser.add_argument("--save_path", help="Name of save directory", default="results")
    parser.add_argument(
        "--data_type",
        type=str,
        choices=["Sri-Lanka", "Kazakhstan", "Weedy-Rice"],
        help="Which dataset should be used",
        required=True,
    )
    parser.add_argument(
        "--model",
        help="Which model to use, and path to the model from project dir, default=model_final.pkl",
        required=True,
    )
    parser.add_argument("--save_images", help="Save the images predicted", action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--device",
        type=str,
        choices=[None, "cpu", "cuda"],
        default=None,
        help="Device for inference. Default=None (auto detects). Use cuda/auto for GPU.",
    )

    args = parser.parse_args()

    run(
        root_dir=args.data_path,
        data_type=args.data_type,
        save_dir=args.save_path,
        single=bool(args.single),
        single_picture=args.jpg,
        amount=args.amount,
        model_path=args.model,
        save_images=bool(args.save_images),
        device_str=args.device,
    )
