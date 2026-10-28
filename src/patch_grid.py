"""Patch-grid helpers shared by the metadata builders: manifests, footprints, and grid indices."""

from __future__ import annotations

import glob
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

MAX_GRID_ROUNDING_ERROR = 0.05
# The patcher derives level-0 magnification as 10 / MPP and falls back to the objective power.
MICRONS_PER_PIXEL_AT_10X = 10.0
MPP_X_PROPERTY = "openslide.mpp-x"
OBJECTIVE_POWER_PROPERTY = "openslide.objective-power"


def load_manifest_shards(pattern: str, columns: Sequence[str]) -> pd.DataFrame:
    """Concatenate the patch manifest shards matching ``pattern``, keeping ``columns``.

    Raises:
        FileNotFoundError: If no shard matches.
    """
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No patch manifest matches {pattern}")
    return pd.concat([pd.read_csv(path, usecols=list(columns)) for path in paths], ignore_index=True)


def compute_patch_footprint(*, patch_size: int, slide_magnification: float, patch_magnification: float) -> float:
    """Return the side of a patch in level-0 pixels."""
    return patch_size * slide_magnification / patch_magnification


def compute_grid_coordinates(*, level0_coords: np.ndarray, footprint: float) -> np.ndarray:
    """Convert ``(N, 2)`` level-0 offsets of a contiguous patch grid to integer grid indices.

    Raises:
        ValueError: If offsets are not (close to) integer multiples of the footprint.
    """
    scaled = level0_coords / footprint
    grid = np.rint(scaled).astype(np.int64)
    error = float(np.abs(scaled - grid).max(initial=0.0))
    if error > MAX_GRID_ROUNDING_ERROR:
        raise ValueError(f"Patch offsets are off the grid by up to {error:.3f} patches")
    return grid


def get_base_magnification(properties: Mapping[str, str]) -> float:
    """Return a slide's level-0 magnification from its OpenSlide properties, as the patcher does.

    Returns:
        ``10 / mpp_x`` when the MPP is known, otherwise the objective power.

    Raises:
        ValueError: If the slide has neither property.
    """
    mpp = properties.get(MPP_X_PROPERTY)
    if mpp:
        return MICRONS_PER_PIXEL_AT_10X / float(mpp)
    objective_power = properties.get(OBJECTIVE_POWER_PROPERTY)
    if objective_power:
        return float(objective_power)
    raise ValueError("Slide has neither an MPP nor an objective power")


def read_base_magnification(slide_path: Path) -> float:
    """Open a whole-slide image and return its level-0 magnification (see ``get_base_magnification``)."""
    import openslide

    with openslide.OpenSlide(str(slide_path)) as slide:
        return get_base_magnification(dict(slide.properties))
