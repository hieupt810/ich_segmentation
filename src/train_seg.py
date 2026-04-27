"""Phase 1 trainer: segmentation pathways only, AMP + FocalDice, classification head frozen."""

import logging
from typing import Any

import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.dataset import make_loader
from src.loss import BinaryFocalDiceLoss
from src.model import build_model
from src.smote import build_train_dataset
from src.utils import autocast_context, set_seed, setup_logging


class SegmentationTrainer:
    """Trains a UNet++ on binary hemorrhage segmentation while freezing the classification head."""

    def __init__(
        self,
        cfg: Config,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        logger: logging.Logger,
    ) -> None:
        self.cfg = cfg
        self.logger = logger
        self.device = torch.device(cfg.device)

        self.model = model.to(self.device)
        self._freeze_classification_head()

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, cfg.epochs)
        )
        self.loss_fn = BinaryFocalDiceLoss(cfg).to(self.device)
        self.scaler = torch.amp.GradScaler(
            device=self.device.type, enabled=self.device.type == "cuda"
        )

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.best_iou: float = -1.0

    # ------------------------------------------------------------------
    # Architecture surgery
    # ------------------------------------------------------------------
    def _freeze_classification_head(self) -> None:
        """Disable gradients on the classification head; segmentation pathways stay trainable."""
        head = getattr(self.model, "classification_head", None)
        if head is None:
            self.logger.warning("Model has no classification_head; nothing to freeze.")
            return
        for p in head.parameters():
            p.requires_grad = False
        head.eval()
        n_frozen = sum(p.numel() for p in head.parameters())
        self.logger.info("Froze %d parameters in classification_head.", n_frozen)

    # ------------------------------------------------------------------
    # Train / validate steps
    # ------------------------------------------------------------------
    def _step_seg_logits(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass returning only the segmentation logits."""
        out = self.model(images)
        return out[0] if isinstance(out, tuple) else out

    def train_one_epoch(self, epoch: int) -> float:
        """Run one training epoch over ``train_loader`` and return mean loss."""
        self.model.train()
        self.model.classification_head.eval()  # keep BN/dropout frozen too
        total = 0.0
        n_batches = 0

        for images, masks, _labels in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True).float().unsqueeze(1)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast_context(self.device):
                seg_logits = self._step_seg_logits(images)
                loss = self.loss_fn(seg_logits, masks)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total += loss.item()
            n_batches += 1

        mean_loss = total / max(1, n_batches)
        self.logger.info("[Epoch %d] train_loss=%.4f", epoch, mean_loss)
        return mean_loss

    @torch.no_grad()
    def validate(self, epoch: int) -> tuple[float, float, float]:
        """Compute mean validation loss plus per-image IoU and Dice averaged across the val set."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        tp_all, fp_all, fn_all, tn_all = [], [], [], []

        for images, masks, _labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True).float().unsqueeze(1)

            with autocast_context(self.device):
                seg_logits = self._step_seg_logits(images)
                loss = self.loss_fn(seg_logits, masks)

            preds = (torch.sigmoid(seg_logits) > 0.5).long()
            tp, fp, fn, tn = smp.metrics.get_stats(preds, masks.long(), mode="binary")
            tp_all.append(tp)
            fp_all.append(fp)
            fn_all.append(fn)
            tn_all.append(tn)
            total_loss += loss.item()
            n_batches += 1

        tp = torch.cat(tp_all)
        fp = torch.cat(fp_all)
        fn = torch.cat(fn_all)
        tn = torch.cat(tn_all)
        iou = float(
            smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        )
        dice = float(
            smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise")
        )
        mean_loss = total_loss / max(1, n_batches)
        self.logger.info(
            "[Epoch %d] val_loss=%.4f val_iou=%.4f val_dice=%.4f",
            epoch,
            mean_loss,
            iou,
            dice,
        )
        return mean_loss, iou, dice

    # ------------------------------------------------------------------
    # Checkpointing + main loop
    # ------------------------------------------------------------------
    def _save_best(self, iou: float) -> None:
        """Persist the model weights when a new best validation IoU is achieved."""
        self.cfg.output.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.cfg.seg_checkpoint_path)
        self.logger.info(
            "Saved best segmentation checkpoint (IoU=%.4f) to %s",
            iou,
            self.cfg.seg_checkpoint_path,
        )

    def _load_best(self) -> None:
        """Reload the best segmentation checkpoint into the in-memory model."""
        ckpt = self.cfg.seg_checkpoint_path
        if not ckpt.exists():
            self.logger.warning("No checkpoint at %s to reload.", ckpt)
            return
        state = torch.load(ckpt, map_location=self.device)
        self.model.load_state_dict(state)
        self.logger.info("Reloaded best segmentation weights from %s", ckpt)

    def fit(self) -> dict[str, Any]:
        """Run training for ``cfg.epochs`` and reload the best weights before returning."""
        history: list[dict[str, float]] = []
        for epoch in range(1, self.cfg.epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            val_loss, val_iou, val_dice = self.validate(epoch)
            self.scheduler.step()

            if val_iou > self.best_iou:
                self.best_iou = val_iou
                self._save_best(val_iou)

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_iou": val_iou,
                    "val_dice": val_dice,
                }
            )

        self._load_best()
        return {"best_iou": self.best_iou, "history": history}


def run_phase1(cfg: Config, logger: logging.Logger | None = None) -> torch.nn.Module:
    """Wire dataset (+SMOTE), model, and trainer; return the model with best weights loaded."""
    logger = logger or setup_logging()
    set_seed(cfg.seed)
    cfg.output.mkdir(parents=True, exist_ok=True)

    train_dataset = build_train_dataset(cfg, logger=logger)
    train_loader = make_loader(
        cfg, phase="train", logger=logger, shuffle=True, dataset=train_dataset
    )
    val_loader = make_loader(cfg, phase="val", logger=logger, shuffle=False)

    model = build_model(
        encoder_name=cfg.encoder_name,
        encoder_weights=cfg.encoder_weights,
        num_clf_classes=cfg.num_clf_classes,
    )
    trainer = SegmentationTrainer(cfg, model, train_loader, val_loader, logger)
    summary = trainer.fit()
    logger.info("Phase 1 complete. Best IoU=%.4f", summary["best_iou"])
    return trainer.model
