from __future__ import annotations

from pathlib import Path

import torch


def build_model(in_channels: int):
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise RuntimeError(
            "segmentation_models_pytorch is required. Install `segmentation-models-pytorch` before training."
        ) from exc

    cache_dir = Path("outputs/cache/torch").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(cache_dir))

    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=1,
        activation=None,
    )
