"""Defines the model architecture for segmentation and classification tasks."""

from typing import Any

import segmentation_models_pytorch as smp


def build_model(
    architecture: str,
    encoder_name: str,
    encoder_weights: str | None,
    num_clf_classes: int,
    *,
    num_seg_classes: int = 1,
    in_channels: int = 3,
    dropout: float = 0.2,
    pooling: str = "avg",
) -> smp.UnetPlusPlus:
    """Builds a UNet++ model for segmentation and classification."""
    aux_params: dict[str, Any] = {
        "activation": None,
        "classes": num_clf_classes,
        "dropout": dropout,
        "pooling": pooling,
    }
    return getattr(smp, architecture)(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_seg_classes,
        aux_params=aux_params,
    )
