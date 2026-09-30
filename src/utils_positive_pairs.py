"""Select augmentation and spatial views from SSL dataset batches."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from torch import Tensor


Batch = Sequence[object]


def two_distortion_positive_pairs(batch: Batch) -> tuple[Tensor, Tensor]:
    """Return the two augmented views produced by the standard dataset."""
    views = cast(Sequence[Tensor], batch[0])
    return views[0], views[1]


def three_views_positive_pairs(batch: Batch) -> tuple[Tensor, Tensor, Tensor]:
    """Return two anchor views and one spatial-neighbor view."""
    views = cast(Sequence[Tensor], batch[0])
    return views[0], views[1], views[2]
