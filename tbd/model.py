"""
Model factory for the segmentation experiment.

Uses segmentation_models_pytorch (smp) for U-Net with ResNet-18 encoder.
"""

import segmentation_models_pytorch as smp


def build_unet(
    in_channels: int = 3,
    num_classes: int = 3,
    encoder_name: str = "resnet18",
    encoder_weights: str = None,
) -> smp.Unet:
    """
    Build a U-Net with ResNet-18 encoder.

    Args:
        in_channels: Number of input channels.
            - 3 for RGB
            - N for multispectral (depends on your band count)
        num_classes: Number of segmentation classes.
            - 3 for weedy rice (background, rice, weedy rice)
        encoder_name: Encoder backbone. Keep 'resnet18' for this experiment.
        encoder_weights: Pretrained weights. Set to None because:
            1. ImageNet weights expect 3 channels — incompatible with multispectral
            2. Using pretrained weights for RGB but not multispectral would introduce an unfair comparison
            3. Training from scratch ensures the ONLY variable is the input data

    Returns:
        smp.Unet model
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,  # Raw logits — we apply softmax in loss/inference
    )
    return model
