import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from src.config import Config


class BinaryFocalDiceLoss(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.focal_weight: float = cfg.focal_weight
        self.dice_weight: float = cfg.dice_weight

        self.focal = smp.losses.FocalLoss(
            mode="binary", alpha=0.25, gamma=2.0, normalized=True
        )
        self.dice = smp.losses.DiceLoss(
            mode="binary", from_logits=True, smooth=1.0, eps=1e-7
        )

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        focal_loss = self.focal(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return self.focal_weight * focal_loss + self.dice_weight * dice_loss
