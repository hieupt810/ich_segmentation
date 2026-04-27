"""Dataset, transforms, and DataLoader factory for ICH multi-window CT slices."""

import logging
from pathlib import Path
from typing import Literal

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

from src.config import Config
from src.utils import seed_worker, setup_logging


def derive_class_label(class_names: list[str], row: pd.Series) -> int:
    """Return 0 for 'no hemorrhage' or the 1-indexed subtype found in the row."""
    for i, col in enumerate(class_names):
        if row[col] == 1:
            return i + 1
    return 0


class ICHDataset(Dataset):
    """Stacks the configured CT windows into a 3-channel tensor with binary mask + class label."""

    def __init__(
        self,
        cfg: Config,
        phase: Literal["train", "val"],
        transform: A.Compose | None = None,
        logger: logging.Logger | None = None,
        return_meta: bool = False,
    ):
        logger = logger or setup_logging()

        self.cfg = cfg
        self.transform = transform or build_val_transforms(cfg)
        self.return_meta = return_meta

        csv_path: Path = cfg.root / cfg.labels_csv
        df: pd.DataFrame = pd.read_csv(csv_path)
        df = df[df["Phase"] == phase].reset_index(drop=True)
        if df.empty:
            logger.warning(
                "No rows found for phase '%s' in CSV file: %s", phase, csv_path
            )

        images_root: Path = cfg.root / cfg.images_dir
        if not images_root.exists():
            raise FileNotFoundError(f"Images directory not found: {images_root}")
        masks_root: Path = cfg.root / cfg.masks_dir
        if not masks_root.exists():
            raise FileNotFoundError(f"Masks directory not found: {masks_root}")

        self.image_paths: list[list[Path]] = []
        self.mask_paths: list[Path] = []
        self.labels: list[int] = []
        self.patient_ids: list[str] = []
        self.slice_ids: list[str] = []
        for _, row in df.iterrows():
            patient: str = f"{int(row['PatientNumber']):03d}"
            slice_idx: str = f"{int(row['SliceNumber']) - 1:04d}"

            # Construct paths for all windows and the mask
            images = [
                images_root / patient / f"{slice_idx}_{window}.png"
                for window in cfg.windows
            ]
            mask = masks_root / patient / f"{slice_idx}.png"

            # Check if all image paths and the mask path exist
            missing_files = [str(p) for p in [*images, mask] if not p.exists()]
            if missing_files:
                logger.warning(
                    "Missing files for row %d: %s", row.name, ", ".join(missing_files)
                )
                continue

            # If all files exist, add them to the dataset
            self.image_paths.append(images)
            self.mask_paths.append(mask)
            self.labels.append(derive_class_label(cfg.class_names, row))
            self.patient_ids.append(patient)
            self.slice_ids.append(slice_idx)

        logger.info(
            "Initialized ICHDataset with %d samples across phase '%s'",
            len(self.labels),
            phase,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        """Return (image, mask, label) plus (patient_id, slice_id) when return_meta is True."""
        channels: list[np.ndarray] = [
            cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in self.image_paths[index]
        ]
        image: np.ndarray = np.stack(channels, axis=-1)

        mask: np.ndarray = cv2.imread(str(self.mask_paths[index]), cv2.IMREAD_GRAYSCALE)
        mask = (mask / 255.0).astype(np.float32)

        augmented = self.transform(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]

        if self.return_meta:
            return (
                image,
                mask,
                self.labels[index],
                self.patient_ids[index],
                self.slice_ids[index],
            )
        return image, mask, self.labels[index]


def build_train_transforms(cfg: Config) -> A.Compose:
    """Augmentations for training: spatial + photometric noise, then normalize + tensor."""
    return A.Compose(
        [
            A.Resize(cfg.image_size, cfg.image_size),
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent=(-0.05, 0.05),
                rotate=(-10, 10),
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5,
            ),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.5),
            A.Normalize(mean=cfg.mean, std=cfg.std),
            ToTensorV2(),
        ]
    )


def build_val_transforms(cfg: Config) -> A.Compose:
    """Deterministic resize + normalize + tensor conversion for validation/inference."""
    return A.Compose(
        [
            A.Resize(cfg.image_size, cfg.image_size),
            A.Normalize(mean=cfg.mean, std=cfg.std),
            ToTensorV2(),
        ]
    )


def make_loader(
    cfg: Config,
    phase: Literal["train", "val"],
    *,
    logger: logging.Logger | None = None,
    shuffle: bool = False,
    dataset: Dataset | None = None,
    return_meta: bool = False,
) -> DataLoader:
    """Build a DataLoader for the requested phase, optionally wrapping a custom dataset."""
    if dataset is None:
        transform = (
            build_train_transforms(cfg)
            if phase == "train"
            else build_val_transforms(cfg)
        )
        dataset = ICHDataset(
            cfg,
            phase,
            transform=transform,
            logger=logger,
            return_meta=return_meta,
        )

    generator = torch.Generator()
    generator.manual_seed(cfg.seed)

    loader_kwargs: dict = {
        "batch_size": cfg.batch_size,
        "shuffle": shuffle,
        "num_workers": cfg.workers,
        "pin_memory": cfg.device == "cuda",
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if cfg.device == "cuda":
        loader_kwargs["pin_memory_device"] = cfg.device
    return DataLoader(dataset, **loader_kwargs)
