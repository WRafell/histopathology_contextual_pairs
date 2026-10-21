"""Frozen-encoder patch embeddings stored per slide."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from src.data import make_coordinate_array

MANIFEST_FILE = "manifest.csv"
SLIDE_DIRECTORY = "slides"
MISSING_PATCH_LABEL = -1


@dataclass(frozen=True)
class SlideEmbeddings:
    """Embeddings and metadata of every patch of one slide, in metadata row order."""

    embeddings: Tensor
    coords: Tensor
    patch_labels: Tensor
    slide_label: int


@torch.no_grad()
def encode_images(
    *, encoder: nn.Module, loader: DataLoader[Tensor], device: torch.device, use_mixed_precision: bool
) -> Tensor:
    """Encode every batch of a loader and return float16 CPU embeddings of shape ``(N, D)``."""
    encoder.eval()
    outputs: list[Tensor] = []
    for images in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_mixed_precision):
            outputs.append(encoder(images.to(device, non_blocking=True)).half().cpu())
    return torch.cat(outputs)


def make_slide_embeddings(*, slide_frame: pd.DataFrame, embeddings: Tensor) -> SlideEmbeddings:
    """Bundle a slide's embeddings with its coordinates and labels (missing patch labels become -1).

    Raises:
        ValueError: If row counts differ or the slide has several slide labels.
    """
    if len(slide_frame) != len(embeddings):
        raise ValueError(f"{len(slide_frame)} patches but {len(embeddings)} embeddings")
    slide_labels = slide_frame["slide_label"].unique()
    if len(slide_labels) != 1:
        raise ValueError(f"Slide has several slide labels: {slide_labels}")
    return SlideEmbeddings(
        embeddings=embeddings,
        coords=torch.from_numpy(make_coordinate_array(slide_frame)),
        patch_labels=torch.as_tensor(slide_frame["label"].fillna(MISSING_PATCH_LABEL).to_numpy(), dtype=torch.long),
        slide_label=int(slide_labels[0]),
    )


def save_slide_embeddings(path: Path, slide: SlideEmbeddings) -> None:
    """Write one slide's embeddings as a ``torch.save`` dictionary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embeddings": slide.embeddings,
            "coords": slide.coords,
            "patch_labels": slide.patch_labels,
            "slide_label": slide.slide_label,
        },
        path,
    )


def load_slide_embeddings(path: Path) -> SlideEmbeddings:
    """Read one slide written by ``save_slide_embeddings``."""
    stored = torch.load(path, map_location="cpu", weights_only=True)
    return SlideEmbeddings(
        embeddings=stored["embeddings"],
        coords=stored["coords"],
        patch_labels=stored["patch_labels"],
        slide_label=int(stored["slide_label"]),
    )


def load_manifest(embedding_dir: Path) -> pd.DataFrame:
    """Read the embedding manifest (``slide, subset, slide_label, path``) with absolute paths.

    Raises:
        FileNotFoundError: If the manifest is missing.
    """
    manifest_path = embedding_dir / MANIFEST_FILE
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Embedding manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    manifest["path"] = [str(embedding_dir / path) for path in manifest["path"]]
    return manifest
