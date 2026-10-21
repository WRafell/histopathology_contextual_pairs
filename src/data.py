"""Patch metadata loading and validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

METADATA_DIRECTORY = "csv"
METADATA_SUFFIX = "_data.csv"
REQUIRED_COLUMNS = frozenset({"img_path", "subset", "slide", "slide_label", "label", "grid_x", "grid_y"})
SUBSETS = ("train", "val", "test")


def resolve_image_path(image_path: str, data_root: Path) -> str:
    """Resolve a metadata image path relative to the data root."""
    path = Path(image_path)
    return str(path if path.is_absolute() else data_root / path)


def load_patch_metadata(dataset: str, data_root: Path) -> pd.DataFrame:
    """Load ``<data_root>/csv/<dataset>_data.csv`` with resolved image paths.

    ``grid_x``/``grid_y`` are integer patch-grid coordinates (adjacent patches differ by 1).
    The returned frame also has ``slide_id``, the integer code of ``slide``.

    Raises:
        FileNotFoundError: If the metadata file is missing.
        ValueError: If required columns or subsets are missing.
    """
    csv_path = data_root / METADATA_DIRECTORY / f"{dataset.lower()}{METADATA_SUFFIX}"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset metadata not found: {csv_path}")
    frame = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
    unknown_subsets = set(frame["subset"]).difference(SUBSETS)
    if unknown_subsets:
        raise ValueError(f"{csv_path} has unknown subsets: {sorted(unknown_subsets)}")

    frame["img_path"] = [resolve_image_path(path, data_root) for path in frame["img_path"]]
    frame["slide_id"] = pd.factorize(frame["slide"])[0]
    return frame


def make_coordinate_array(frame: pd.DataFrame) -> np.ndarray:
    """Return ``grid_x``/``grid_y`` as an ``(N, 2)`` int64 array."""
    return frame[["grid_x", "grid_y"]].to_numpy(dtype=np.int64)


def select_anchor_rows(*, frame: pd.DataFrame, max_anchors: int | None, seed: int) -> np.ndarray:
    """Return sorted row positions of training patches, optionally subsampled without replacement."""
    train_positions = np.flatnonzero(frame["subset"].to_numpy() == "train")
    if max_anchors is None or max_anchors >= len(train_positions):
        return train_positions
    generator = np.random.default_rng(seed)
    return np.sort(generator.choice(train_positions, size=max_anchors, replace=False))
