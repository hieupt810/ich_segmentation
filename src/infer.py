"""Phase 2 evaluation: per-patient segmentation stats + classification metrics + JSON dump."""

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from src.config import Config
from src.dataset import make_loader
from src.model import build_model
from src.utils import autocast_context, setup_logging


class Evaluator:
    """Computes segmentation (per-slice + per-patient) and classification metrics."""

    def __init__(
        self,
        cfg: Config,
        model: torch.nn.Module,
        val_loader: DataLoader,
        logger: logging.Logger,
    ) -> None:
        self.cfg = cfg
        self.logger = logger
        self.device = torch.device(cfg.device)
        self.model = model.to(self.device).eval()
        self.val_loader = val_loader

    @torch.no_grad()
    def run(self) -> dict:
        """Iterate validation set, accumulate metrics, then aggregate per-patient + overall."""
        per_slice_dice: list[float] = []
        per_slice_iou: list[float] = []
        patient_dice: dict[str, list[float]] = defaultdict(list)
        patient_iou: dict[str, list[float]] = defaultdict(list)

        # Running pixel-level confusion for the segmentation 2x2 matrix.
        global_tp = global_fp = global_fn = global_tn = 0

        clf_preds: list[int] = []
        clf_targets: list[int] = []

        for batch in self.val_loader:
            images, masks, labels, patient_ids, _slice_ids = batch
            images = images.to(self.device, non_blocking=True)
            target_masks = masks.to(self.device, non_blocking=True).float().unsqueeze(1)
            labels = labels.to(self.device, non_blocking=True).long()

            with autocast_context(self.device):
                seg_logits, clf_logits = self.model(images)

            pred_masks = (torch.sigmoid(seg_logits) > 0.5).long()
            tp, fp, fn, tn = smp.metrics.get_stats(
                pred_masks, target_masks.long(), mode="binary"
            )
            dice_per_slice = smp.metrics.f1_score(tp, fp, fn, tn, reduction=None)
            iou_per_slice = smp.metrics.iou_score(tp, fp, fn, tn, reduction=None)

            global_tp += int(tp.sum().item())
            global_fp += int(fp.sum().item())
            global_fn += int(fn.sum().item())
            global_tn += int(tn.sum().item())

            for i, pid in enumerate(patient_ids):
                d = float(dice_per_slice[i].item())
                u = float(iou_per_slice[i].item())
                per_slice_dice.append(d)
                per_slice_iou.append(u)
                patient_dice[pid].append(d)
                patient_iou[pid].append(u)

            clf_preds.extend(clf_logits.argmax(dim=1).cpu().tolist())
            clf_targets.extend(labels.cpu().tolist())

        seg_metrics = self._summarize_segmentation(
            per_slice_dice,
            per_slice_iou,
            patient_dice,
            patient_iou,
            global_tp,
            global_fp,
            global_fn,
            global_tn,
        )
        clf_metrics = self._summarize_classification(clf_targets, clf_preds)
        return {"segmentation": seg_metrics, "classification": clf_metrics}

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------
    def _summarize_segmentation(
        self,
        per_slice_dice: list[float],
        per_slice_iou: list[float],
        patient_dice: dict[str, list[float]],
        patient_iou: dict[str, list[float]],
        tp: int,
        fp: int,
        fn: int,
        tn: int,
    ) -> dict:
        """Aggregate slice-level + per-patient + overall pixel statistics."""
        per_patient: dict[str, dict[str, float]] = {}
        patient_means_d: list[float] = []
        patient_means_u: list[float] = []
        for pid in sorted(patient_dice.keys()):
            d = np.asarray(patient_dice[pid], dtype=np.float64)
            u = np.asarray(patient_iou[pid], dtype=np.float64)
            per_patient[pid] = {
                "mean_dice": float(d.mean()),
                "std_dice": float(d.std(ddof=0)),
                "mean_iou": float(u.mean()),
                "std_iou": float(u.std(ddof=0)),
                "n_slices": int(d.size),
            }
            patient_means_d.append(float(d.mean()))
            patient_means_u.append(float(u.mean()))

        slice_d = np.asarray(per_slice_dice, dtype=np.float64)
        slice_u = np.asarray(per_slice_iou, dtype=np.float64)
        pat_d = np.asarray(patient_means_d, dtype=np.float64)
        pat_u = np.asarray(patient_means_u, dtype=np.float64)

        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        denom = 2 * tp + fp + fn
        overall_dice = float((2 * tp) / denom) if denom > 0 else 0.0
        denom_iou = tp + fp + fn
        overall_iou = float(tp / denom_iou) if denom_iou > 0 else 0.0

        overall = {
            "mean_dice_slice": float(slice_d.mean()) if slice_d.size else 0.0,
            "std_dice_slice": float(slice_d.std(ddof=0)) if slice_d.size else 0.0,
            "mean_iou_slice": float(slice_u.mean()) if slice_u.size else 0.0,
            "std_iou_slice": float(slice_u.std(ddof=0)) if slice_u.size else 0.0,
            "mean_dice_patient": float(pat_d.mean()) if pat_d.size else 0.0,
            "std_dice_patient": float(pat_d.std(ddof=0)) if pat_d.size else 0.0,
            "mean_iou_patient": float(pat_u.mean()) if pat_u.size else 0.0,
            "std_iou_patient": float(pat_u.std(ddof=0)) if pat_u.size else 0.0,
            "pixel_dice": overall_dice,
            "pixel_iou": overall_iou,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "confusion_matrix": [[tn, fp], [fn, tp]],
        }
        return {"overall": overall, "per_patient": per_patient}

    def _summarize_classification(self, targets: list[int], preds: list[int]) -> dict:
        """Macro F1 + per-class precision/recall/specificity + confusion matrix."""
        labels = list(range(self.cfg.num_clf_classes))
        cm = confusion_matrix(targets, preds, labels=labels)
        macro_f1 = float(f1_score(targets, preds, average="macro", zero_division=0))
        precision, _, f1_per, support = precision_recall_fscore_support(
            targets, preds, labels=labels, zero_division=0
        )

        cm_arr = cm.astype(np.int64)
        total = int(cm_arr.sum())
        per_class: dict[str, dict[str, float]] = {}
        names = ["NoHemorrhage", *self.cfg.class_names]
        for cls_idx, name in enumerate(names):
            tp = int(cm_arr[cls_idx, cls_idx])
            fn = int(cm_arr[cls_idx, :].sum() - tp)
            fp = int(cm_arr[:, cls_idx].sum() - tp)
            tn = int(total - tp - fn - fp)
            sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            per_class[name] = {
                "precision": float(precision[cls_idx]),
                "sensitivity": sens,
                "specificity": spec,
                "f1": float(f1_per[cls_idx]),
                "support": int(support[cls_idx]),
            }

        return {
            "macro_f1": macro_f1,
            "per_class": per_class,
            "confusion_matrix": cm_arr.tolist(),
            "labels": names,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, metrics: dict, path: Path | None = None) -> Path:
        """Serialize metrics to JSON and log a compact summary."""
        out = path or (self.cfg.output / "metrics.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(metrics, f, indent=2)

        seg = metrics["segmentation"]["overall"]
        clf = metrics["classification"]
        self.logger.info(
            "[Eval] seg patient mean Dice=%.4f±%.4f IoU=%.4f±%.4f | clf macro F1=%.4f",
            seg["mean_dice_patient"],
            seg["std_dice_patient"],
            seg["mean_iou_patient"],
            seg["std_iou_patient"],
            clf["macro_f1"],
        )
        self.logger.info("Saved metrics to %s", out)
        return out


def run_evaluation(
    cfg: Config,
    logger: logging.Logger | None = None,
) -> dict:
    """Top-level evaluation entry: load best clf checkpoint, compute metrics, write JSON."""
    logger = logger or setup_logging()
    cfg.output.mkdir(parents=True, exist_ok=True)

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
    logger.info("Loaded checkpoint for evaluation: %s", ckpt)

    evaluator = Evaluator(cfg, model, val_loader, logger)
    metrics = evaluator.run()
    evaluator.save(metrics)
    return metrics
