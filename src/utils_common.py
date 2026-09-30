"""Shared reproducibility and run naming helpers."""

from __future__ import annotations

import random

import numpy as np
import torch
from omegaconf import DictConfig


def set_manual_seed(random_seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)


def get_basename(args: DictConfig) -> str:
    """Build a stable checkpoint name from the selected method settings."""
    dataset = str(args.dataset).lower()
    method = str(args.positive_pair_sampling)
    framework = str(args.ssl_method).lower()
    backbone = str(args.backbone_name).lower()
    if method == "closest_patch":
        return (
            f"ds={dataset}_ssl={framework}_pps={method}"
            f"-max={args.maximum_distance}-min={args.minimum_distance}"
            f"_alpha={args.alpha}_encoder={backbone}"
        )
    return f"ds={dataset}_ssl={framework}_pps={method}_encoder={backbone}"
