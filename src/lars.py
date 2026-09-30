"""LARS optimizer and its step-based learning-rate schedule."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import torch
from torch import Tensor


class AdjustLearningRate:
    """Set a linear-warmup, cosine-decay learning rate before each step."""

    def __init__(self, *, max_epochs: int, batch_size: int, base_lr: float, num_batches: int) -> None:
        if min(max_epochs, batch_size, num_batches) < 1:
            raise ValueError("max_epochs, batch_size, and num_batches must be positive")
        self.max_steps = max_epochs * num_batches
        self.warmup_steps = min(10 * num_batches, self.max_steps)
        self.base_lr = base_lr * batch_size / 256

    def step(self, optimizer: torch.optim.Optimizer, global_step: int) -> float:
        """Update all optimizer groups and return the current learning rate."""
        if not 0 <= global_step < self.max_steps:
            raise ValueError(f"global_step must be in [0, {self.max_steps})")
        if global_step < self.warmup_steps:
            learning_rate = self.base_lr * global_step / max(1, self.warmup_steps)
        else:
            decay_steps = max(1, self.max_steps - self.warmup_steps)
            progress = (global_step - self.warmup_steps) / decay_steps
            end_lr = self.base_lr * 0.001
            learning_rate = end_lr + 0.5 * (self.base_lr - end_lr) * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        return learning_rate


class LARS(torch.optim.Optimizer):
    """Layer-wise adaptive rate scaling with momentum."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float,
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        eta: float = 0.001,
        weight_decay_filter: Callable[[Tensor], bool] | None = None,
        lars_adaptation_filter: Callable[[Tensor], bool] | None = None,
    ) -> None:
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "eta": eta,
            "weight_decay_filter": weight_decay_filter,
            "lars_adaptation_filter": lars_adaptation_filter,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                update = parameter.grad
                if group["weight_decay_filter"] is None or not group["weight_decay_filter"](parameter):
                    update = update.add(parameter, alpha=group["weight_decay"])
                if group["lars_adaptation_filter"] is None or not group["lars_adaptation_filter"](parameter):
                    parameter_norm = torch.norm(parameter)
                    update_norm = torch.norm(update)
                    scale = torch.where(
                        (parameter_norm > 0) & (update_norm > 0),
                        group["eta"] * parameter_norm / update_norm,
                        torch.ones_like(parameter_norm),
                    )
                    update = update * scale

                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(parameter)
                momentum_buffer = state["momentum_buffer"]
                momentum_buffer.mul_(group["momentum"]).add_(update)
                parameter.add_(momentum_buffer, alpha=-group["lr"])
        return loss


def exclude_bias_and_norm(parameter: Tensor) -> bool:
    """Return whether LARS should skip decay/adaptation for a parameter."""
    return parameter.ndim == 1
