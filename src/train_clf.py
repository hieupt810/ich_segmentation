"""Phase 2 trainer: classification-head fine-tune with encoder/decoder/seg-head frozen."""

import logging
from collections import Counter
from typing import Any

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.config import Config
from src.dataset import make_loader
from src.model import build_model
from src.utils import autocast_context, set_seed, setup_logging


class ClassificationTrainer:
    """Fine-tunes only the classification head; segmentation pathways stay frozen and intact."""

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
        self._load_seg_checkpoint()
        self._freeze_segmentation()

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable, lr=cfg.clf_learning_rate, weight_decay=cfg.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, cfg.clf_epochs)
        )
        self.loss_fn = nn.CrossEntropyLoss(
            weight=self._compute_class_weights(train_loader)
        ).to(self.device)
        self.scaler = torch.amp.GradScaler(
            device=self.device.type, enabled=self.device.type == "cuda"
        )

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.best_f1: float = -1.0

    # ------------------------------------------------------------------
    # Architecture surgery
    # ------------------------------------------------------------------
    def _load_seg_checkpoint(self) -> None:
        """Load the Phase 1 segmentation weights into the model before freezing."""
        ckpt = self.cfg.seg_checkpoint_path
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Segmentation checkpoint not found at {ckpt}. Run Phase 1 first."
            )
        state = torch.load(ckpt, map_location=self.device)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            self.logger.warning("Missing keys when loading seg checkpoint: %s", missing)
        if unexpected:
            self.logger.warning(
                "Unexpected keys when loading seg checkpoint: %s", unexpected
            )
        self.logger.info("Loaded segmentation checkpoint from %s", ckpt)

    def _freeze_segmentation(self) -> None:
        """Disable gradients on encoder, decoder, and segmentation head."""
        for module_name in ("encoder", "decoder", "segmentation_head"):
            module = getattr(self.model, module_name, None)
            if module is None:
                self.logger.warning("Model has no '%s' attribute.", module_name)
                continue
            for p in module.parameters():
                p.requires_grad = False
            module.eval()

        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_clf = sum(p.numel() for p in self.model.classification_head.parameters())
        self.logger.info(
            "Trainable params: %d (classification_head=%d). Match expected.",
            n_trainable,
            n_clf,
        )
        if n_trainable != n_clf:
            raise RuntimeError(
                "Freeze mismatch: trainable params differ from classification_head size."
            )

    # ------------------------------------------------------------------
    # Class weights
    # ------------------------------------------------------------------
    def _compute_class_weights(self, loader: DataLoader) -> torch.Tensor:
        """Inverse-frequency weights derived from the training-set label distribution."""
        ds = loader.dataset
        labels = getattr(ds, "labels", None)
        if labels is None:
            self.logger.warning(
                "Dataset exposes no 'labels'; using uniform class weights."
            )
            return torch.ones(self.cfg.num_clf_classes, device=self.device)

        counts = Counter(int(label) for label in labels)
        weights = torch.ones(self.cfg.num_clf_classes, dtype=torch.float32)
        total = sum(counts.values())
        for cls in range(self.cfg.num_clf_classes):
            c = counts.get(cls, 0)
            weights[cls] = total / (self.cfg.num_clf_classes * c) if c > 0 else 1.0
        self.logger.info("Class weights: %s", weights.tolist())
        return weights.to(self.device)

    # ------------------------------------------------------------------
    # Train / validate steps
    # ------------------------------------------------------------------
    def _step_clf_logits(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass returning only the classification logits."""
        out = self.model(images)
        return out[1] if isinstance(out, tuple) else out

    def train_one_epoch(self, epoch: int) -> float:
        """Run one training epoch updating only classification-head weights."""
        # Keep frozen pathways in eval mode (BN/dropout pinned).
        self.model.eval()
        self.model.classification_head.train()

        total = 0.0
        n_batches = 0
        for images, _masks, labels in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).long()

            self.optimizer.zero_grad(set_to_none=True)
            with autocast_context(self.device):
                logits = self._step_clf_logits(images)
                loss = self.loss_fn(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total += loss.item()
            n_batches += 1

        mean_loss = total / max(1, n_batches)
        self.logger.info("[CLF Epoch %d] train_loss=%.4f", epoch, mean_loss)
        return mean_loss

    @torch.no_grad()
    def validate(self, epoch: int) -> tuple[float, float]:
        """Compute mean validation loss and macro F1 over all 6 classes."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds: list[int] = []
        all_targets: list[int] = []

        for images, _masks, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).long()

            with autocast_context(self.device):
                logits = self._step_clf_logits(images)
                loss = self.loss_fn(logits, labels)

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(labels.cpu().tolist())
            total_loss += loss.item()
            n_batches += 1

        macro_f1 = float(
            f1_score(all_targets, all_preds, average="macro", zero_division=0)
        )
        mean_loss = total_loss / max(1, n_batches)
        self.logger.info(
            "[CLF Epoch %d] val_loss=%.4f val_macro_f1=%.4f", epoch, mean_loss, macro_f1
        )
        return mean_loss, macro_f1

    # ------------------------------------------------------------------
    # Checkpointing + main loop
    # ------------------------------------------------------------------
    def _save_best(self, f1: float) -> None:
        """Persist the full model state when a new best macro F1 is reached."""
        self.cfg.output.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.cfg.clf_checkpoint_path)
        self.logger.info(
            "Saved best classification checkpoint (F1=%.4f) to %s",
            f1,
            self.cfg.clf_checkpoint_path,
        )

    def _load_best(self) -> None:
        """Reload the best classification checkpoint into the in-memory model."""
        ckpt = self.cfg.clf_checkpoint_path
        if not ckpt.exists():
            self.logger.warning("No checkpoint at %s to reload.", ckpt)
            return
        state = torch.load(ckpt, map_location=self.device)
        self.model.load_state_dict(state)
        self.logger.info("Reloaded best classification weights from %s", ckpt)

    def fit(self) -> dict[str, Any]:
        """Run training for ``cfg.clf_epochs`` and reload best weights before returning."""
        history: list[dict[str, float]] = []
        for epoch in range(1, self.cfg.clf_epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            val_loss, val_f1 = self.validate(epoch)
            self.scheduler.step()

            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self._save_best(val_f1)

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_macro_f1": val_f1,
                }
            )

        self._load_best()
        return {"best_macro_f1": self.best_f1, "history": history}


def run_phase2(cfg: Config, logger: logging.Logger | None = None) -> torch.nn.Module:
    """Wire dataset, model, and trainer for Phase 2; return model with best clf weights."""
    logger = logger or setup_logging()
    set_seed(cfg.seed)
    cfg.output.mkdir(parents=True, exist_ok=True)

    train_loader = make_loader(cfg, phase="train", logger=logger, shuffle=True)
    val_loader = make_loader(cfg, phase="val", logger=logger, shuffle=False)

    model = build_model(
        encoder_name=cfg.encoder_name,
        encoder_weights=cfg.encoder_weights,
        num_clf_classes=cfg.num_clf_classes,
    )
    trainer = ClassificationTrainer(cfg, model, train_loader, val_loader, logger)
    summary = trainer.fit()
    logger.info("Phase 2 complete. Best macro F1=%.4f", summary["best_macro_f1"])
    return trainer.model
