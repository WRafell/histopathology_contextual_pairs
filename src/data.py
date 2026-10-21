"""Patch metadata loading and validation."""

from __future__ import annotations

from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd

METADATA_DIRECTORY = "csv"
METADATA_SUFFIX = "_data.csv"
REQUIRED_COLUMNS = frozenset({"img_path", "subset", "slide", "slide_label", "label"})
SUBSETS = ("train", "val", "test")

# Camelyon16 patch filenames start with level-0 pixel offsets "<x>_<y>_..."; patches of
# 256 px at 20x span 512 level-0 pixels, so dividing by the stride gives grid coordinates.
CAMELYON16_DATASET = "camelyon16"
CAMELYON16_GRID_STRIDE = 512


def compute_camelyon16_coordinates(image_path: str) -> tuple[int, int]:
    """Return grid coordinates parsed from a Camelyon16 patch filename.

    Raises:
        ValueError: If the filename does not start with two integers.
    """
    x_value, y_value = map(int, Path(image_path).stem.split("_")[:2])
    return x_value // CAMELYON16_GRID_STRIDE, y_value // CAMELYON16_GRID_STRIDE


def parse_coordinates(value: object) -> tuple[int, int]:
    """Parse a coordinate pair stored as a tuple, list, or its string form.

    Raises:
        ValueError: If the value is not a pair.
    """
    parsed = literal_eval(value) if isinstance(value, str) else value
    if not isinstance(parsed, (tuple, list)) or len(parsed) != 2:
        raise ValueError(f"Expected a coordinate pair, got {value!r}")
    return int(parsed[0]), int(parsed[1])


def resolve_image_path(image_path: str, data_root: Path) -> str:
    """Resolve a metadata image path relative to the data root."""
    path = Path(image_path)
    return str(path if path.is_absolute() else data_root / path)


def load_patch_metadata(dataset: str, data_root: Path) -> pd.DataFrame:
    """Load ``<data_root>/csv/<dataset>_data.csv`` with resolved paths and grid coordinates.

    The returned frame has the required columns plus ``coords`` (tuple of ints) and
    ``slide_id`` (integer code of ``slide``).

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
    if dataset.lower() == CAMELYON16_DATASET:
        frame["coords"] = [compute_camelyon16_coordinates(path) for path in frame["img_path"]]
    elif "coords" in frame:
        frame["coords"] = [parse_coordinates(value) for value in frame["coords"]]
    else:
        raise ValueError(f"{csv_path} needs a coords column")
    frame["slide_id"] = pd.factorize(frame["slide"])[0]
    return frame


def make_coordinate_array(frame: pd.DataFrame) -> np.ndarray:
    """Return the ``coords`` column as an ``(N, 2)`` int64 array."""
    return np.asarray(frame["coords"].tolist(), dtype=np.int64).reshape(-1, 2)


def select_anchor_rows(*, frame: pd.DataFrame, max_anchors: int | None, seed: int) -> np.ndarray:
    """Return sorted row positions of training patches, optionally subsampled without replacement."""
    train_positions = np.flatnonzero(frame["subset"].to_numpy() == "train")
    if max_anchors is None or max_anchors >= len(train_positions):
        return train_positions
    generator = np.random.default_rng(seed)
    return np.sort(generator.choice(train_positions, size=max_anchors, replace=False))
