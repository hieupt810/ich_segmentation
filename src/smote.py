"""SMOTE-based oversampling for ICH slices that preserves anatomical mask integrity."""

import logging
from typing import Literal

import albumentations as A
import cv2
import numpy as np
from imblearn.over_sampling import SMOTE
from torch.utils.data import Dataset

from src.config import Config
from src.dataset import ICHDataset, build_train_transforms
from src.utils import setup_logging


class SmotedICHDataset(Dataset):
    """In-memory dataset of original + SMOTE-synthesized slices with train-time transforms."""

    def __init__(
        self,
        images: np.ndarray,
        masks: np.ndarray,
        labels: np.ndarray,
        transform: A.Compose,
    ) -> None:
        self.images = images
        self.masks = masks
        self.labels = labels.astype(np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        """Apply training-time augmentations on the fly for original and synthetic samples."""
        image = self.images[index]
        mask = self.masks[index]
        augmented = self.transform(image=image, mask=mask)
        return augmented["image"], augmented["mask"], int(self.labels[index])


def _load_raw(
    dataset: ICHDataset, image_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and resize raw (un-augmented) image/mask pairs into stacked numpy arrays."""
    n = len(dataset)
    c = len(dataset.cfg.windows)
    images = np.empty((n, image_size, image_size, c), dtype=np.float32)
    masks = np.empty((n, image_size, image_size), dtype=np.float32)
    labels = np.empty((n,), dtype=np.int64)

    for i in range(n):
        channels = [
            cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in dataset.image_paths[i]
        ]
        image = np.stack(channels, axis=-1).astype(np.float32) / 255.0
        mask = cv2.imread(str(dataset.mask_paths[i]), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        image = cv2.resize(
            image, (image_size, image_size), interpolation=cv2.INTER_AREA
        )
        mask = cv2.resize(
            mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST
        )

        images[i] = image
        masks[i] = mask
        labels[i] = dataset.labels[i]
    return images, masks, labels


def apply_smote(
    base: ICHDataset,
    cfg: Config,
    logger: logging.Logger | None = None,
) -> SmotedICHDataset:
    """Oversample minority hemorrhage classes with SMOTE on flattened image+mask vectors."""
    logger = logger or setup_logging()

    images, masks, labels = _load_raw(base, cfg.image_size)
    h, w, c = cfg.image_size, cfg.image_size, len(cfg.windows)

    # Flatten each sample to a single vector: image first, mask appended at the end.
    image_dim = h * w * c
    mask_dim = h * w
    flat = np.concatenate(
        [images.reshape(len(images), image_dim), masks.reshape(len(masks), mask_dim)],
        axis=1,
    ).astype(np.float32)

    # Counter-style log of class balance pre-SMOTE.
    unique, counts = np.unique(labels, return_counts=True)
    logger.info(
        "Pre-SMOTE class counts: %s",
        dict(zip(unique.tolist(), counts.tolist(), strict=False)),
    )

    minority_count = counts.min()
    k_neighbors = max(1, min(cfg.smote_neighbors, minority_count - 1))
    if k_neighbors < cfg.smote_neighbors:
        logger.warning(
            "Reducing SMOTE k_neighbors to %d due to minority class size %d",
            k_neighbors,
            minority_count,
        )

    smote = SMOTE(k_neighbors=k_neighbors, random_state=cfg.seed)
    flat_resampled, labels_resampled = smote.fit_resample(flat, labels)

    images_out = flat_resampled[:, :image_dim].reshape(-1, h, w, c)
    masks_out = flat_resampled[:, image_dim:].reshape(-1, h, w)

    # SMOTE interpolations produce continuous mask values; re-binarize at 0.5.
    masks_out = (masks_out > 0.5).astype(np.float32)
    images_out = np.clip(images_out, 0.0, 1.0).astype(np.float32)

    # Albumentations expects uint8 RGB-like arrays for default augmentations.
    images_uint8 = (images_out * 255.0).astype(np.uint8)

    unique, counts = np.unique(labels_resampled, return_counts=True)
    logger.info(
        "Post-SMOTE class counts: %s",
        dict(zip(unique.tolist(), counts.tolist(), strict=False)),
    )

    return SmotedICHDataset(
        images=images_uint8,
        masks=masks_out,
        labels=labels_resampled,
        transform=build_train_transforms(cfg),
    )


def build_train_dataset(
    cfg: Config,
    logger: logging.Logger | None = None,
    *,
    phase: Literal["train"] = "train",
) -> Dataset:
    """Build the training dataset; applies SMOTE when ``cfg.use_smote`` is True."""
    logger = logger or setup_logging()
    base = ICHDataset(
        cfg, phase=phase, transform=build_train_transforms(cfg), logger=logger
    )
    if not cfg.use_smote:
        return base
    return apply_smote(base, cfg, logger)
