from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import Tensor, nn

from src.datasets import PositivePairDataset
from src.ssl_models import compute_positive_pair_loss


class MeanSquaredErrorModel:
    """Identity encoder with a mean-squared-error pair loss."""

    def __init__(self) -> None:
        self.backbone: nn.Module = nn.Identity()

    def encode(self, view: Tensor) -> tuple[Tensor, Tensor]:
        return view, view

    def compute_pair_loss(self, first: tuple[Tensor, Tensor], second: tuple[Tensor, Tensor]) -> Tensor:
        return ((first[0] - second[0]) ** 2).mean()


def test_two_view_loss_ignores_context() -> None:
    model = MeanSquaredErrorModel()
    anchor, augmented, context = torch.zeros(2, 3), torch.ones(2, 3), torch.full((2, 3), 3.0)
    two_view = compute_positive_pair_loss(
        model=model, anchor_view=anchor, augmented_view=augmented, context_view=None, context_weight=0.0
    )
    three_view = compute_positive_pair_loss(
        model=model, anchor_view=anchor, augmented_view=augmented, context_view=context, context_weight=0.5
    )
    assert float(two_view) == 1.0
    assert float(three_view) == 5.0
    with pytest.raises(ValueError):
        compute_positive_pair_loss(
            model=model, anchor_view=anchor, augmented_view=augmented, context_view=None, context_weight=0.5
        )


def test_dataset_returns_context_view_only_when_rows_are_given(tmp_path: Path) -> None:
    paths = []
    for index in range(2):
        path = tmp_path / f"{index}.png"
        Image.fromarray(np.full((8, 8, 3), index * 100, dtype=np.uint8)).save(path)
        paths.append(str(path))

    def to_tensor(image: Image.Image) -> Tensor:
        return torch.from_numpy(np.asarray(image).copy())

    anchors = np.array([0])
    common = {"image_paths": paths, "anchor_rows": anchors, "first_transform": to_tensor, "second_transform": to_tensor}
    assert len(PositivePairDataset(context_rows=None, **common)[0]) == 2
    anchor_view, _, context_view = PositivePairDataset(context_rows=np.array([1]), **common)[0]
    assert int(anchor_view.max()) == 0
    assert int(context_view.max()) == 100
