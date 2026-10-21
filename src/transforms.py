"""Image transforms for SSL views and frozen-encoder evaluation."""

from __future__ import annotations

import torch
from torchvision.transforms import InterpolationMode, v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BLUR_KERNEL_SIZE = (5, 9)
BLUR_SIGMA = (0.1, 5.0)
COLOR_JITTER_PROBABILITY = 0.5
BLUR_PROBABILITY = 0.2
FLIP_PROBABILITY = 0.5


def make_view_transform(*, image_size: int, min_crop_scale: float, grayscale_probability: float) -> v2.Compose:
    """Return a stochastic SSL augmentation producing a normalized ``image_size`` tensor."""
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(min_crop_scale, 1.0),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            v2.RandomHorizontalFlip(p=FLIP_PROBABILITY),
            v2.RandomVerticalFlip(p=FLIP_PROBABILITY),
            v2.RandomApply(
                [v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=COLOR_JITTER_PROBABILITY
            ),
            v2.RandomGrayscale(p=grayscale_probability),
            v2.RandomApply([v2.GaussianBlur(kernel_size=BLUR_KERNEL_SIZE, sigma=BLUR_SIGMA)], p=BLUR_PROBABILITY),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def make_view_transforms(*, image_size: int) -> tuple[v2.Compose, v2.Compose]:
    """Return the asymmetric first and second SSL view transforms."""
    first = make_view_transform(image_size=image_size, min_crop_scale=0.8, grayscale_probability=0.1)
    second = make_view_transform(image_size=image_size, min_crop_scale=0.6, grayscale_probability=0.2)
    return first, second


def make_eval_transform(*, image_size: int) -> v2.Compose:
    """Return the deterministic resize-and-normalize transform for frozen encoders."""
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(image_size, image_size), interpolation=InterpolationMode.BICUBIC, antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
