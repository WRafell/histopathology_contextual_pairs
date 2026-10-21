"""Slide-level classifiers over frozen patch embeddings and their training loop."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional

from src.config import Aggregator


@dataclass(frozen=True)
class Bag:
    """Patch embeddings of one slide and its label."""

    embeddings: Tensor
    label: int


class AttentionMIL(nn.Module):
    """Attention-based MIL pooling (Ilse et al., 2018) followed by a linear classifier."""

    def __init__(self, *, embedding_dim: int, attention_dim: int, num_classes: int) -> None:
        super().__init__()
        self.attention: nn.Sequential = nn.Sequential(
            nn.Linear(embedding_dim, attention_dim), nn.Tanh(), nn.Linear(attention_dim, 1)
        )
        self.classifier: nn.Linear = nn.Linear(embedding_dim, num_classes)

    def forward(self, embeddings: Tensor) -> Tensor:
        """Map ``(N, D)`` patch embeddings to ``(1, C)`` slide logits."""
        weights = torch.softmax(self.attention(embeddings).T, dim=1)
        return self.classifier(weights @ embeddings)


class MeanPoolingMIL(nn.Module):
    """Average patch embeddings, then apply a linear classifier (aggregation-free control)."""

    def __init__(self, *, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        self.classifier: nn.Linear = nn.Linear(embedding_dim, num_classes)

    def forward(self, embeddings: Tensor) -> Tensor:
        """Map ``(N, D)`` patch embeddings to ``(1, C)`` slide logits."""
        return self.classifier(embeddings.mean(dim=0, keepdim=True))


def make_mil_model(*, aggregator: Aggregator, embedding_dim: int, attention_dim: int, num_classes: int) -> nn.Module:
    """Build the slide classifier named by ``aggregator``.

    Raises:
        ValueError: If the aggregator is unsupported.
    """
    if aggregator is Aggregator.ABMIL:
        return AttentionMIL(embedding_dim=embedding_dim, attention_dim=attention_dim, num_classes=num_classes)
    if aggregator is Aggregator.MEAN_POOLING:
        return MeanPoolingMIL(embedding_dim=embedding_dim, num_classes=num_classes)
    raise ValueError(f"Unsupported aggregator: {aggregator}")


@torch.no_grad()
def predict_bags(*, model: nn.Module, bags: Sequence[Bag], device: torch.device) -> tuple[np.ndarray, float]:
    """Return ``(N, C)`` class probabilities and the mean cross-entropy over bags."""
    model.eval()
    probabilities: list[Tensor] = []
    total_loss = 0.0
    for bag in bags:
        logits = model(bag.embeddings.to(device).float())
        total_loss += float(functional.cross_entropy(logits, torch.tensor([bag.label], device=device)))
        probabilities.append(torch.softmax(logits, dim=1).cpu())
    return torch.cat(probabilities).numpy(), total_loss / len(bags)


def train_mil_model(
    *,
    model: nn.Module,
    train_bags: Sequence[Bag],
    validation_bags: Sequence[Bag],
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[nn.Module, list[dict[str, float]]]:
    """Train on one bag per step and return the copy with the lowest validation loss.

    Returns:
        The best model and per-epoch ``train_loss``/``validation_loss`` records.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    best_model, best_loss = copy.deepcopy(model), float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        for index in torch.randperm(len(train_bags), generator=generator).tolist():
            bag = train_bags[index]
            logits = model(bag.embeddings.to(device).float())
            loss = functional.cross_entropy(logits, torch.tensor([bag.label], device=device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
        scheduler.step()
        _, validation_loss = predict_bags(model=model, bags=validation_bags, device=device)
        history.append(
            {"epoch": epoch + 1, "train_loss": total_loss / len(train_bags), "validation_loss": validation_loss}
        )
        if validation_loss < best_loss:
            best_model, best_loss = copy.deepcopy(model), validation_loss
    return best_model, history
