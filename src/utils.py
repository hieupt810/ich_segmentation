from __future__ import annotations

import logging
import os
import random
import sys

import numpy as np
import torch


# -----------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------
def setup_logging() -> logging.Logger:
    """Configure logging to output to stdout with a specific format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    return logging.getLogger()


# ---------------------------------------------------------------------------
# Autocast settings
# ---------------------------------------------------------------------------
def autocast_settings(device: torch.device) -> tuple[bool, torch.dtype, str]:
    """Return autocast settings based on the device type."""
    return (
        (True, torch.float16, "cuda")
        if device.type == "cuda"
        else (False, torch.float32, "cpu")
    )


def autocast_context(device: torch.device) -> torch.amp.autocast:
    """Return the appropriate autocast context manager based on the device type."""
    return (
        torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
        if device.type == "cuda"
        else torch.amp.autocast(device_type="cpu", dtype=torch.float32, enabled=False)
    )


# ---------------------------------------------------------------------------
# Seed setting
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Set the random seed for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(_: int) -> None:
    """Seed each DataLoader worker deterministically."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
