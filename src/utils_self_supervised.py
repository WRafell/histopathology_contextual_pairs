"""Datasets for self-supervised learning."""

from __future__ import annotations

from ast import literal_eval
from collections.abc import Callable, Sequence
from typing import cast
from pathlib import Path
import random

from PIL import Image
import torch
from torch import Tensor


TransformPair = Callable[[Image.Image], tuple[Tensor, Tensor]]


def _load_rgb_image(image_path: str) -> Image.Image:
    """Load an image and detach it from its open file handle."""
    with Image.open(image_path) as image:
        return image.convert("RGB")


def _parse_neighbor_buckets(value: str | Sequence[Sequence[str]]) -> list[list[str]]:
    """Parse distance-indexed neighbor paths from a CSV cell."""
    buckets = literal_eval(value) if isinstance(value, str) else value
    if not isinstance(buckets, (tuple, list)):
        raise ValueError("Neighbor metadata must be a list of distance buckets")
    return [list(bucket) for bucket in buckets]


class BasicSSLDataset(torch.utils.data.Dataset[tuple[tuple[Tensor, Tensor], str]]):
    """Load patches and return two augmented views for SSL training."""

    def __init__(self, data: Sequence[Sequence[str]], transforms: TransformPair) -> None:
        self.image_paths = [row[0] for row in data]
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[tuple[Tensor, Tensor], str]:
        image_path = self.image_paths[index]
        views = self.transforms(_load_rgb_image(image_path))
        return views, image_path


class NeighborDataset(torch.utils.data.Dataset[tuple[tuple[Tensor, Tensor, Tensor], tuple[str, str]]]):
    """Load anchors and sample same-slide neighbors from precomputed distance buckets."""

    def __init__(
        self,
        data: Sequence[Sequence[str]],
        minimum_distance: int,
        maximum_distance: int | None,
        transforms: TransformPair,
    ) -> None:
        if minimum_distance < 0:
            raise ValueError("minimum_distance cannot be negative")
        if maximum_distance is not None and maximum_distance < minimum_distance:
            raise ValueError("maximum_distance must be at least minimum_distance")

        self.image_paths = [row[0] for row in data]
        self.transforms = transforms
        self.neighbors: list[list[str]] = []
        for image_path, raw_buckets in data:
            buckets = _parse_neighbor_buckets(raw_buckets)
            end = len(buckets) if maximum_distance is None else maximum_distance + 1
            selected = [path for bucket in buckets[minimum_distance:end] for path in bucket]
            self.neighbors.append(selected or [image_path])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[tuple[Tensor, Tensor, Tensor], tuple[str, str]]:
        anchor_path = self.image_paths[index]
        neighbor_path = random.choice(self.neighbors[index])
        anchor_view, second_anchor_view = self.transforms(_load_rgb_image(anchor_path))
        neighbor_view, _ = self.transforms(_load_rgb_image(neighbor_path))
        return (anchor_view, second_anchor_view, neighbor_view), (anchor_path, neighbor_path)
