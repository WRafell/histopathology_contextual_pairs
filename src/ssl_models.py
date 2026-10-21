"""SSL model wrappers: an encoder plus heads, and the factory that builds them."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Protocol

import torch
from torch import Tensor, nn

from src.config import SSLFramework, SSLTrainConfig
from src.ssl_losses import combine_pair_losses, compute_barlow_twins_loss, compute_byol_loss, compute_vicreg_loss

# Each encoded view is (online output, target output); they are the same tensor for
# frameworks without a target network.
EncodedView = tuple[Tensor, Tensor]


class SSLModel(Protocol):
    """Interface shared by the SSL wrappers."""

    backbone: nn.Module

    def encode(self, view: Tensor) -> EncodedView: ...

    def compute_pair_loss(self, first: EncodedView, second: EncodedView) -> Tensor: ...


def make_projector(*, input_dim: int, layer_dims: Sequence[int]) -> nn.Sequential:
    """Return ``Linear-BN-ReLU`` hidden blocks followed by a bias-free output ``Linear``."""
    dims = [input_dim, *layer_dims]
    layers: list[nn.Module] = []
    for in_dim, out_dim in zip(dims[:-2], dims[1:-1], strict=True):
        layers += [nn.Linear(in_dim, out_dim, bias=False), nn.BatchNorm1d(out_dim), nn.ReLU(inplace=True)]
    layers.append(nn.Linear(dims[-2], dims[-1], bias=False))
    return nn.Sequential(*layers)


class VICRegModel(nn.Module):
    """Encoder with a VICReg expander."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        embedding_dim: int,
        projector_dims: Sequence[int],
        invariance_weight: float,
        variance_weight: float,
        covariance_weight: float,
    ) -> None:
        super().__init__()
        self.backbone: nn.Module = backbone
        self.projector: nn.Sequential = make_projector(input_dim=embedding_dim, layer_dims=projector_dims)
        self.invariance_weight: float = invariance_weight
        self.variance_weight: float = variance_weight
        self.covariance_weight: float = covariance_weight

    def encode(self, view: Tensor) -> EncodedView:
        projection = self.projector(self.backbone(view))
        return projection, projection

    def compute_pair_loss(self, first: EncodedView, second: EncodedView) -> Tensor:
        return compute_vicreg_loss(
            first=first[0],
            second=second[0],
            invariance_weight=self.invariance_weight,
            variance_weight=self.variance_weight,
            covariance_weight=self.covariance_weight,
        )


class BarlowTwinsModel(nn.Module):
    """Encoder with a Barlow Twins projector."""

    def __init__(
        self, *, backbone: nn.Module, embedding_dim: int, projector_dims: Sequence[int], off_diagonal_weight: float
    ) -> None:
        super().__init__()
        self.backbone: nn.Module = backbone
        self.projector: nn.Sequential = make_projector(input_dim=embedding_dim, layer_dims=projector_dims)
        self.off_diagonal_weight: float = off_diagonal_weight

    def encode(self, view: Tensor) -> EncodedView:
        projection = self.projector(self.backbone(view))
        return projection, projection

    def compute_pair_loss(self, first: EncodedView, second: EncodedView) -> Tensor:
        return compute_barlow_twins_loss(first=first[0], second=second[0], off_diagonal_weight=self.off_diagonal_weight)


class BYOLModel(nn.Module):
    """Online encoder, projector, and predictor with an EMA target network."""

    def __init__(
        self, *, backbone: nn.Module, embedding_dim: int, hidden_dim: int, projection_dim: int, target_momentum: float
    ) -> None:
        super().__init__()
        self.backbone: nn.Module = backbone
        self.projector: nn.Sequential = make_projector(input_dim=embedding_dim, layer_dims=(hidden_dim, projection_dim))
        self.predictor: nn.Sequential = make_projector(
            input_dim=projection_dim, layer_dims=(hidden_dim, projection_dim)
        )
        self.target_backbone: nn.Module = copy.deepcopy(backbone).requires_grad_(False)
        self.target_projector: nn.Sequential = copy.deepcopy(self.projector).requires_grad_(False)
        self.target_momentum: float = target_momentum

    def encode(self, view: Tensor) -> EncodedView:
        prediction = self.predictor(self.projector(self.backbone(view)))
        with torch.no_grad():
            target = self.target_projector(self.target_backbone(view))
        return prediction, target

    def compute_pair_loss(self, first: EncodedView, second: EncodedView) -> Tensor:
        return compute_byol_loss(prediction=first[0], target=second[1]) + compute_byol_loss(
            prediction=second[0], target=first[1]
        )

    @torch.no_grad()
    def update_target(self) -> None:
        """Move target weights toward online weights in place (EMA with ``target_momentum``)."""
        pairs = (
            (self.backbone, self.target_backbone),
            (self.projector, self.target_projector),
        )
        for online, target in pairs:
            for online_parameter, target_parameter in zip(online.parameters(), target.parameters(), strict=True):
                target_parameter.lerp_(online_parameter, 1.0 - self.target_momentum)


def make_ssl_model(*, config: SSLTrainConfig, backbone: nn.Module, embedding_dim: int) -> nn.Module:
    """Build the SSL wrapper selected by ``config.framework``.

    Raises:
        ValueError: If the framework is unsupported.
    """
    if config.framework is SSLFramework.VICREG:
        return VICRegModel(
            backbone=backbone,
            embedding_dim=embedding_dim,
            projector_dims=config.vicreg.projector_dims,
            invariance_weight=config.vicreg.invariance_weight,
            variance_weight=config.vicreg.variance_weight,
            covariance_weight=config.vicreg.covariance_weight,
        )
    if config.framework is SSLFramework.BARLOW_TWINS:
        return BarlowTwinsModel(
            backbone=backbone,
            embedding_dim=embedding_dim,
            projector_dims=config.barlow_twins.projector_dims,
            off_diagonal_weight=config.barlow_twins.off_diagonal_weight,
        )
    if config.framework is SSLFramework.BYOL:
        return BYOLModel(
            backbone=backbone,
            embedding_dim=embedding_dim,
            hidden_dim=config.byol.hidden_dim,
            projection_dim=config.byol.projection_dim,
            target_momentum=config.byol.target_momentum,
        )
    raise ValueError(f"Unsupported SSL framework: {config.framework}")


def compute_positive_pair_loss(
    *, model: SSLModel, anchor_view: Tensor, augmented_view: Tensor, context_view: Tensor, context_weight: float
) -> Tensor:
    """Return ``(1 - alpha) L(anchor, augmented) + alpha L(anchor, context)``, encoding each view once."""
    anchor = model.encode(anchor_view)
    augmentation_loss = model.compute_pair_loss(anchor, model.encode(augmented_view))
    context_loss = model.compute_pair_loss(anchor, model.encode(context_view))
    return combine_pair_losses(
        augmentation_loss=augmentation_loss, context_loss=context_loss, context_weight=context_weight
    )
