"""Command-line interface for the multi-task ICH pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.config import Config


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser with subcommands for each pipeline stage."""
    parser = argparse.ArgumentParser(
        prog="ich-pipeline",
        description="DenseNet201 UNet++ multi-task segmentation + classification pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_seg = subparsers.add_parser(
        "train-seg",
        help="Phase 1: train segmentation pathways (classification head frozen).",
    )
    _add_shared_overrides(train_seg)
    train_seg.set_defaults(func=_cmd_train_seg)

    train_clf = subparsers.add_parser(
        "train-clf",
        help="Phase 2: fine-tune the classification head (segmentation pathways frozen).",
    )
    _add_shared_overrides(train_clf)
    train_clf.set_defaults(func=_cmd_train_clf)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Phase 2: compute metrics + write 4-panel inference grids.",
    )
    _add_shared_overrides(evaluate)
    evaluate.add_argument(
        "--max-vis-samples",
        type=int,
        default=None,
        help="Cap the number of inference grids saved (default: save every validation slice).",
    )
    evaluate.set_defaults(func=_cmd_evaluate)

    run_all = subparsers.add_parser(
        "all", help="Run train-seg, train-clf, and evaluate sequentially."
    )
    _add_shared_overrides(run_all)
    run_all.add_argument("--max-vis-samples", type=int, default=None)
    run_all.set_defaults(func=_cmd_all)

    return parser


def _add_shared_overrides(p: argparse.ArgumentParser) -> None:
    """Attach the override flags shared across every subcommand."""
    p.add_argument(
        "--epochs", type=int, default=None, help="Override seg-phase epochs."
    )
    p.add_argument(
        "--clf-epochs", type=int, default=None, help="Override clf-phase epochs."
    )
    p.add_argument("--lr", type=float, default=None, help="Override seg learning rate.")
    p.add_argument(
        "--clf-lr", type=float, default=None, help="Override clf learning rate."
    )
    p.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    p.add_argument(
        "--workers", type=int, default=None, help="Override DataLoader workers."
    )
    p.add_argument(
        "--weight-decay", type=float, default=None, help="Override AdamW weight decay."
    )
    p.add_argument(
        "--smote-neighbors", type=int, default=None, help="Override SMOTE k_neighbors."
    )
    p.add_argument(
        "--no-smote",
        dest="use_smote",
        action="store_false",
        default=None,
        help="Disable SMOTE oversampling for the segmentation phase.",
    )
    p.add_argument("--seed", type=int, default=None, help="Override random seed.")
    p.add_argument(
        "--image-size", type=int, default=None, help="Override input resolution."
    )
    p.add_argument(
        "--encoder-name", type=str, default=None, help="Override SMP encoder name."
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override output directory (checkpoints, metrics, grids).",
    )


# ---------------------------------------------------------------------------
# Override application
# ---------------------------------------------------------------------------
_OVERRIDE_MAP = {
    "epochs": "epochs",
    "clf_epochs": "clf_epochs",
    "lr": "learning_rate",
    "clf_lr": "clf_learning_rate",
    "batch_size": "batch_size",
    "workers": "workers",
    "weight_decay": "weight_decay",
    "smote_neighbors": "smote_neighbors",
    "use_smote": "use_smote",
    "seed": "seed",
    "image_size": "image_size",
    "encoder_name": "encoder_name",
    "output": "output",
}


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    """Mutate ``cfg`` with any non-None overrides parsed from the CLI."""
    for arg_name, cfg_attr in _OVERRIDE_MAP.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            setattr(cfg, cfg_attr, value)
    return cfg


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------
def _cmd_train_seg(
    cfg: Config, logger: logging.Logger, _args: argparse.Namespace
) -> None:
    """Run Phase 1 training."""
    from src.train_seg import run_phase1

    run_phase1(cfg, logger)


def _cmd_train_clf(
    cfg: Config, logger: logging.Logger, _args: argparse.Namespace
) -> None:
    """Run Phase 2 classification fine-tuning."""
    from src.train_clf import run_phase2

    run_phase2(cfg, logger)


def _cmd_evaluate(
    cfg: Config, logger: logging.Logger, args: argparse.Namespace
) -> None:
    """Run evaluation: metrics JSON + 4-panel inference grids."""
    from src.infer import run_evaluation
    from src.visualize import run_inference_artifacts

    run_evaluation(cfg, logger)
    run_inference_artifacts(cfg, logger, max_samples=args.max_vis_samples)


def _cmd_all(cfg: Config, logger: logging.Logger, args: argparse.Namespace) -> None:
    """Run all three phases sequentially."""
    _cmd_train_seg(cfg, logger, args)
    _cmd_train_clf(cfg, logger, args)
    _cmd_evaluate(cfg, logger, args)
