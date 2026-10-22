"""PyTorch datasets that load patch images."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

ImageTransform = Callable[[Image.Image], Tensor]


def load_rgb_image(image_path: str) -> Image.Image:
    """Load an image as RGB and close its file handle."""
    with Image.open(image_path) as image:
        return image.convert("RGB")


class PositivePairDataset(Dataset[tuple[Tensor, ...]]):
    """Return two views of each anchor and, if ``context_rows`` is given, one view of its contextual positive.

    The anchor gets ``first_transform`` and ``second_transform`` (the augmentation pair); the
    contextual patch gets ``second_transform`` so that it plays the role of the second view.
    Without ``context_rows``, items are ``(anchor_view, augmented_view)`` only.
    Augmentation randomness comes from the worker's torch RNG, which the DataLoader seeds.
    """

    def __init__(
        self,
        *,
        image_paths: Sequence[str],
        anchor_rows: np.ndarray,
        context_rows: np.ndarray | None,
        first_transform: ImageTransform,
        second_transform: ImageTransform,
    ) -> None:
        if context_rows is not None and anchor_rows.shape != context_rows.shape:
            raise ValueError("anchor_rows and context_rows must have the same shape")
        self.image_paths: Sequence[str] = image_paths
        self.anchor_rows: np.ndarray = anchor_rows
        self.context_rows: np.ndarray | None = context_rows
        self.first_transform: ImageTransform = first_transform
        self.second_transform: ImageTransform = second_transform

    def __len__(self) -> int:
        return len(self.anchor_rows)

    def __getitem__(self, index: int) -> tuple[Tensor, ...]:
        anchor = load_rgb_image(self.image_paths[self.anchor_rows[index]])
        views = (self.first_transform(anchor), self.second_transform(anchor))
        if self.context_rows is None:
            return views
        context = load_rgb_image(self.image_paths[self.context_rows[index]])
        return (*views, self.second_transform(context))


class ImageDataset(Dataset[Tensor]):
    """Return one deterministic transformed image per path."""

    def __init__(self, *, image_paths: Sequence[str], transform: ImageTransform) -> None:
        self.image_paths: Sequence[str] = image_paths
        self.transform: ImageTransform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tensor:
        return self.transform(load_rgb_image(self.image_paths[index]))
