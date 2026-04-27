"""Project-wide configuration dataclass for the multi-task ICH pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass(slots=True)
class Config:
    """Static defaults for model, data, training, and runtime knobs."""

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    architecture: str = "UnetPlusPlus"
    encoder_name: str = "densenet201"
    encoder_weights: str | None = "imagenet"
    class_names: Sequence[str] = field(
        default_factory=lambda: [
            "Intraventricular",
            "Intraparenchymal",
            "Subarachnoid",
            "Epidural",
            "Subdural",
        ]
    )

    @property
    def num_clf_classes(self) -> int:
        """Number of classification logits (subtypes + 'no hemorrhage')."""
        return len(self.class_names) + 1

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    root: Path = field(default_factory=lambda: Path("data"))
    images_dir: str = "images"
    masks_dir: str = "masks"
    labels_csv: str = "diagnosis_split.csv"
    image_size: int = 224
    windows: Sequence[str] = field(
        default_factory=lambda: [
            "brain_stripped",
            "subdural_stripped",
            "brain_normal",
        ]
    )
    mean: Sequence[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: Sequence[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    focal_weight: float = 0.5
    dice_weight: float = 0.5

    # ------------------------------------------------------------------
    # Training (segmentation phase)
    # ------------------------------------------------------------------
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    workers: int = 4
    seed: int = 42

    # ------------------------------------------------------------------
    # Training (classification phase)
    # ------------------------------------------------------------------
    clf_epochs: int = 30
    clf_learning_rate: float = 1e-4

    # ------------------------------------------------------------------
    # SMOTE
    # ------------------------------------------------------------------
    use_smote: bool = True
    smote_neighbors: int = 3

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------
    @property
    def output(self) -> Path:
        return Path(f"output_{self.encoder_name}")

    @property
    def device(self) -> str:
        """Return 'cuda' when a GPU is available, else 'cpu'."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def seg_checkpoint_path(self) -> Path:
        """Path to the best segmentation-stage checkpoint."""
        return self.output / f"best_{self.encoder_name}_seg.pth"

    @property
    def clf_checkpoint_path(self) -> Path:
        """Path to the best classification-stage checkpoint."""
        return self.output / f"best_{self.encoder_name}_clf.pth"
