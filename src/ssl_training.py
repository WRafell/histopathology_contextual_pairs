"""SSL training loop: one epoch over positive pairs with optional fp16 autocast."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.lars import compute_learning_rate
from src.ssl_models import BYOLModel


@dataclass(frozen=True)
class LearningRateSchedule:
    """Step-wise LARS schedule parameters."""

    total_steps: int
    warmup_steps: int
    peak_learning_rate: float


@dataclass(frozen=True)
class EpochResult:
    """Summary of one training epoch."""

    mean_loss: float
    last_learning_rate: float
    num_skipped_steps: int
    next_step: int


def set_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    """Set the learning rate of every parameter group in place."""
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def train_one_epoch(
    *,
    model: nn.Module,
    loss_module: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, ...]],
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    schedule: LearningRateSchedule,
    first_step: int,
    context_weight: float,
    device: torch.device,
    use_mixed_precision: bool,
) -> EpochResult:
    """Run one epoch; updates ``model``, ``optimizer``, and ``scaler`` in place.

    ``loss_module`` is ``PositivePairLoss(model)``, possibly wrapped in DistributedDataParallel; the
    returned mean loss is this process's own. Batches are ``(anchor, augmented)`` or
    ``(anchor, augmented, context)`` views.

    Steps whose fp16 gradients overflow are skipped by the scaler and counted.
    """
    model.train()
    total_loss = 0.0
    num_skipped_steps = 0
    step = first_step
    learning_rate = 0.0
    for anchor_view, augmented_view, *context_views in loader:
        learning_rate = compute_learning_rate(
            step=step,
            total_steps=schedule.total_steps,
            warmup_steps=schedule.warmup_steps,
            peak_learning_rate=schedule.peak_learning_rate,
        )
        set_learning_rate(optimizer, learning_rate)
        optimizer.zero_grad(set_to_none=True)
        autocast = (
            torch.autocast(device_type=device.type, dtype=torch.float16)
            if use_mixed_precision
            else contextlib.nullcontext()
        )
        with autocast:
            loss = loss_module(
                anchor_view.to(device, non_blocking=True),
                augmented_view.to(device, non_blocking=True),
                context_views[0].to(device, non_blocking=True) if context_views else None,
                context_weight,
            )
        scale_before = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scaler.is_enabled() and scaler.get_scale() < scale_before:
            num_skipped_steps += 1
        if isinstance(model, BYOLModel):
            model.update_target()
        total_loss += float(loss.detach())
        step += 1
    num_batches = step - first_step
    return EpochResult(
        mean_loss=total_loss / max(1, num_batches),
        last_learning_rate=learning_rate,
        num_skipped_steps=num_skipped_steps,
        next_step=step,
    )
