"""LARS optimizer and its warm-up/cosine learning-rate schedule."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import torch
from torch import Tensor

REFERENCE_BATCH_SIZE = 256
FINAL_LEARNING_RATE_FRACTION = 0.001
LARS_MOMENTUM = 0.9
LARS_TRUST_COEFFICIENT = 0.001


def compute_peak_learning_rate(*, base_learning_rate: float, batch_size: int) -> float:
    """Scale the base learning rate linearly with batch size (``base * B / 256``)."""
    return base_learning_rate * batch_size / REFERENCE_BATCH_SIZE


def compute_learning_rate(*, step: int, total_steps: int, warmup_steps: int, peak_learning_rate: float) -> float:
    """Linear warm-up to ``peak_learning_rate``, then cosine decay to 0.1% of it.

    Raises:
        ValueError: If ``step`` is outside ``[0, total_steps)``.
    """
    if not 0 <= step < total_steps:
        raise ValueError(f"step must be in [0, {total_steps}), got {step}")
    if step < warmup_steps:
        return peak_learning_rate * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    final = peak_learning_rate * FINAL_LEARNING_RATE_FRACTION
    return final + 0.5 * (peak_learning_rate - final) * (1 + math.cos(math.pi * progress))


def is_bias_or_norm(parameter: Tensor) -> bool:
    """Return whether a parameter is 1-D (bias or norm), which LARS excludes from decay and adaptation."""
    return parameter.ndim == 1


class LARS(torch.optim.Optimizer):
    """Layer-wise adaptive rate scaling with momentum (You et al., 2017)."""

    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        weight_decay: float,
        is_excluded: Callable[[Tensor], bool] = is_bias_or_norm,
    ) -> None:
        super().__init__(params, {"lr": 0.0, "weight_decay": weight_decay})
        self.is_excluded: Callable[[Tensor], bool] = is_excluded

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # pyrefly: ignore[bad-override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                update = parameter.grad
                if not self.is_excluded(parameter):
                    update = update.add(parameter, alpha=group["weight_decay"])
                    parameter_norm = torch.norm(parameter)
                    update_norm = torch.norm(update)
                    trust_ratio = torch.where(
                        (parameter_norm > 0) & (update_norm > 0),
                        LARS_TRUST_COEFFICIENT * parameter_norm / update_norm,
                        torch.ones_like(parameter_norm),
                    )
                    update = update * trust_ratio
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(parameter)
                momentum_buffer = state["momentum_buffer"]
                momentum_buffer.mul_(LARS_MOMENTUM).add_(update)
                parameter.add_(momentum_buffer, alpha=-group["lr"])
        return loss
