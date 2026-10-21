"""Negative-free SSL objectives as pure tensor functions (computed in float32)."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional

VARIANCE_EPSILON = 1e-4
STANDARDIZE_EPSILON = 1e-5


def compute_off_diagonal(matrix: Tensor) -> Tensor:
    """Return the off-diagonal entries of a square matrix as a flat tensor."""
    size = matrix.shape[0]
    if matrix.shape != (size, size):
        raise ValueError(f"Expected a square matrix, got {tuple(matrix.shape)}")
    return matrix.flatten()[:-1].view(size - 1, size + 1)[:, 1:].flatten()


def compute_vicreg_loss(
    *,
    first: Tensor,
    second: Tensor,
    invariance_weight: float,
    variance_weight: float,
    covariance_weight: float,
) -> Tensor:
    """VICReg loss (Bardes et al., 2022) between two ``(B, D)`` projections."""
    first, second = first.float(), second.float()
    batch_size, num_features = first.shape
    invariance = functional.mse_loss(first, second)
    first = first - first.mean(dim=0)
    second = second - second.mean(dim=0)
    std_first = torch.sqrt(first.var(dim=0) + VARIANCE_EPSILON)
    std_second = torch.sqrt(second.var(dim=0) + VARIANCE_EPSILON)
    variance = functional.relu(1 - std_first).mean() / 2 + functional.relu(1 - std_second).mean() / 2
    cov_first = first.T @ first / (batch_size - 1)
    cov_second = second.T @ second / (batch_size - 1)
    covariance = (
        compute_off_diagonal(cov_first).pow(2).sum() + compute_off_diagonal(cov_second).pow(2).sum()
    ) / num_features
    return invariance_weight * invariance + variance_weight * variance + covariance_weight * covariance


def standardize_batch(features: Tensor) -> Tensor:
    """Standardize each feature over the batch (BatchNorm without affine, training mode)."""
    mean = features.mean(dim=0)
    variance = features.var(dim=0, unbiased=False)
    return (features - mean) / torch.sqrt(variance + STANDARDIZE_EPSILON)


def compute_barlow_twins_loss(*, first: Tensor, second: Tensor, off_diagonal_weight: float) -> Tensor:
    """Barlow Twins loss (Zbontar et al., 2021) between two ``(B, D)`` projections."""
    first, second = first.float(), second.float()
    correlation = standardize_batch(first).T @ standardize_batch(second) / first.shape[0]
    on_diagonal = (torch.diagonal(correlation) - 1).pow(2).sum()
    off_diagonal = compute_off_diagonal(correlation).pow(2).sum()
    return on_diagonal + off_diagonal_weight * off_diagonal


def compute_byol_loss(*, prediction: Tensor, target: Tensor) -> Tensor:
    """Mean normalized squared error ``2 - 2 cos`` between predictions and stop-gradient targets."""
    prediction = functional.normalize(prediction.float(), dim=1)
    target = functional.normalize(target.detach().float(), dim=1)
    return (2 - 2 * (prediction * target).sum(dim=1)).mean()


def combine_pair_losses(*, augmentation_loss: Tensor, context_loss: Tensor, context_weight: float) -> Tensor:
    """Return ``(1 - alpha) * L_aug + alpha * L_ctx``."""
    if not 0.0 <= context_weight <= 1.0:
        raise ValueError(f"context_weight must be in [0, 1], got {context_weight}")
    return (1.0 - context_weight) * augmentation_loss + context_weight * context_loss
