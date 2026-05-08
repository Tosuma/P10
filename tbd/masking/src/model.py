from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


DEFAULT_MODEL_CONFIG = {
    "architecture": "Unet",
    "encoder_name": "resnet34",
    "encoder_weights": "imagenet",
    "classes": 1,
}

ARCHITECTURE_ALIASES = {
    "unet": "Unet",
    "unetplusplus": "UnetPlusPlus",
    "unet++": "UnetPlusPlus",
    "deeplabv3plus": "DeepLabV3Plus",
    "deeplabv3+": "DeepLabV3Plus",
    "segformer": "Segformer",
}


def resolve_model_config(config: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_MODEL_CONFIG)
    if config:
        merged.update(config)
    architecture = str(merged["architecture"])
    merged["architecture"] = ARCHITECTURE_ALIASES.get(architecture.lower(), architecture)
    merged["encoder_name"] = str(merged["encoder_name"])
    merged["encoder_weights"] = merged.get("encoder_weights")
    merged["classes"] = int(merged.get("classes", 1))
    return merged


def build_model(in_channels: int, model_config: dict[str, Any] | None = None):
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise RuntimeError(
            "segmentation_models_pytorch is required. Install `segmentation-models-pytorch` before training."
        ) from exc

    cache_dir = Path("outputs/cache/torch").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(cache_dir))

    resolved = resolve_model_config(model_config)
    architecture = resolved["architecture"]
    model_cls = getattr(smp, architecture, None)
    if model_cls is None:
        if architecture == "Segformer":
            raise RuntimeError(
                "Segformer is not available in the installed segmentation_models_pytorch version. "
                "Install segmentation-models-pytorch>=0.5.0."
            )
        raise RuntimeError(
            f"Unsupported architecture '{architecture}'. "
            "Ensure the config uses a supported segmentation_models_pytorch class name."
        )

    return model_cls(
        encoder_name=resolved["encoder_name"],
        encoder_weights=resolved["encoder_weights"],
        in_channels=in_channels,
        classes=resolved["classes"],
        activation=None,
    )
