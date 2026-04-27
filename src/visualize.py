"""Side-by-side inference grids: original CT, GT mask, predicted mask, heatmap overlay."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.dataset import make_loader
from src.model import build_model
from src.utils import autocast_context, setup_logging


def _denormalize_channel(
    image_chw: torch.Tensor, mean: list[float], std: list[float], channel: int = 2
) -> np.ndarray:
    """Recover a single CT window for display (default: brain_normal channel)."""
    img = image_chw.detach().cpu().float()
    m = torch.tensor(mean).view(-1, 1, 1)
    s = torch.tensor(std).view(-1, 1, 1)
    img = img * s + m
    img = img.clamp(0.0, 1.0)
    return img[channel].numpy()


def save_inference_grid(
    image_chw: torch.Tensor,
    gt_mask: torch.Tensor,
    pred_logits: torch.Tensor,
    save_path: Path,
    cfg: Config,
) -> None:
    """Render and save a 1x4 plot: [Original | GT | Pred | Heatmap overlay]."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    ct = _denormalize_channel(image_chw, list(cfg.mean), list(cfg.std), channel=2)
    gt = gt_mask.detach().cpu().float().numpy()
    probs = torch.sigmoid(pred_logits).detach().cpu().float().numpy().squeeze()
    pred = (probs > 0.5).astype(np.float32)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(ct, cmap="gray")
    axes[0].set_title("Original CT")
    axes[1].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth")
    axes[2].imshow(pred, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Prediction")
    axes[3].imshow(ct, cmap="gray")
    axes[3].imshow(probs, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    axes[3].set_title("Heatmap")
    for ax in axes:
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def save_inference_artifacts(
    cfg: Config,
    model: torch.nn.Module,
    val_loader: DataLoader,
    logger: logging.Logger,
    max_samples: int = 16,
) -> Path:
    """Save up to ``max_samples`` 4-panel inference grids under ``cfg.output / 'inference'``."""
    out_dir = cfg.output / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device)
    model = model.to(device).eval()

    saved = 0
    for batch in val_loader:
        if len(batch) == 5:
            images, masks, _labels, patient_ids, slice_ids = batch
        else:
            images, masks, _labels = batch
            patient_ids = ["?"] * images.shape[0]
            slice_ids = [f"{i:04d}" for i in range(images.shape[0])]

        images = images.to(device, non_blocking=True)
        with autocast_context(device):
            seg_logits, _clf_logits = model(images)
        seg_logits = seg_logits.float()

        for i in range(images.shape[0]):
            if saved >= max_samples:
                logger.info("Saved %d inference grids to %s", saved, out_dir)
                return out_dir
            pid = patient_ids[i]
            sid = slice_ids[i]
            save_path = out_dir / f"{pid}_{sid}.png"
            save_inference_grid(
                image_chw=images[i],
                gt_mask=masks[i],
                pred_logits=seg_logits[i],
                save_path=save_path,
                cfg=cfg,
            )
            saved += 1

    logger.info("Saved %d inference grids to %s", saved, out_dir)
    return out_dir


def run_inference_artifacts(
    cfg: Config, logger: logging.Logger | None = None, max_samples: int = 16
) -> Path:
    """Top-level helper: build model, load best checkpoint, dump grids to ``output/inference``."""
    logger = logger or setup_logging()

    val_loader = make_loader(
        cfg, phase="val", logger=logger, shuffle=False, return_meta=True
    )
    model = build_model(
        encoder_name=cfg.encoder_name,
        encoder_weights=None,
        num_clf_classes=cfg.num_clf_classes,
    )
    ckpt = (
        cfg.clf_checkpoint_path
        if cfg.clf_checkpoint_path.exists()
        else cfg.seg_checkpoint_path
    )
    state = torch.load(ckpt, map_location=cfg.device)
    model.load_state_dict(state, strict=False)
    logger.info("Loaded checkpoint for visualization: %s", ckpt)

    return save_inference_artifacts(cfg, model, val_loader, logger, max_samples)
