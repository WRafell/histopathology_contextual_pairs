"""Build ``camelyon16_data.csv`` from patch manifests, the slide split, and tumor masks.

Usage:
    python -m src.build_camelyon16_metadata --config configs/camelyon16_metadata.yaml \
        --output-dir runs/metadata/camelyon16_20x_224

Writes ``<output-dir>/csv/camelyon16_data.csv``; use ``<output-dir>`` as ``data.data_root``.
"""

from __future__ import annotations

import glob
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.cli import RESOLVED_CONFIG_FILE, make_run_directory, parse_run_arguments
from src.config import Camelyon16MetadataConfig, load_config, save_config
from src.data import METADATA_DIRECTORY, METADATA_SUFFIX

DATASET = "camelyon16"
MAX_GRID_ROUNDING_ERROR = 0.05
MANIFEST_COLUMNS = ["patch_path", "parent_wsi", "x_level0", "y_level0", "patch_width"]


def normalize_slide_name(name: str) -> str:
    """Map ``normal-001`` / ``normal_001.tif`` style names to ``normal_001``."""
    return Path(name).stem.replace("-", "_")


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


def compute_box_fractions(*, mask: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Return the fraction of positive mask pixels inside each box.

    Args:
        mask: ``(H, W)`` boolean mask.
        boxes: ``(N, 4)`` integer ``(x0, y0, x1, y1)`` in mask pixels, end-exclusive; clipped to the mask.

    Returns:
        ``(N,)`` fractions in ``[0, 1]``; empty boxes give 0.
    """
    height, width = mask.shape
    integral = np.zeros((height + 1, width + 1), dtype=np.int64)
    integral[1:, 1:] = mask.astype(np.int64).cumsum(axis=0).cumsum(axis=1)
    x0, x1 = np.clip(boxes[:, 0], 0, width), np.clip(boxes[:, 2], 0, width)
    y0, y1 = np.clip(boxes[:, 1], 0, height), np.clip(boxes[:, 3], 0, height)
    counts = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    areas = (x1 - x0) * (y1 - y0)
    return np.divide(counts, areas, out=np.zeros(len(boxes)), where=areas > 0)


def compute_mask_boxes(*, level0_coords: np.ndarray, footprint: float, downsample: float) -> np.ndarray:
    """Return end-exclusive patch boxes in mask pixels (at least one pixel wide)."""
    start = np.floor(level0_coords / downsample).astype(np.int64)
    end = np.maximum(np.ceil((level0_coords + footprint) / downsample).astype(np.int64), start + 1)
    return np.concatenate([start, end], axis=1)


def read_tumor_mask(*, mask_path: Path, target_downsample: float, tumor_value: int) -> tuple[np.ndarray, float]:
    """Read an annotation mask at the pyramid level closest to ``target_downsample``.

    Returns:
        The boolean mask (pixel value == ``tumor_value``) and the level's downsample relative to level 0.
    """
    import openslide

    with openslide.OpenSlide(str(mask_path)) as slide:
        level = slide.get_best_level_for_downsample(target_downsample)
        size = slide.level_dimensions[level]
        region = np.asarray(slide.read_region((0, 0), level, size))[..., 0]
        return region == tumor_value, float(slide.level_downsamples[level])


def label_slide_patches(
    *,
    slide_patches: pd.DataFrame,
    mask_path: Path,
    footprint: float,
    target_downsample: float,
    tumor_value: int,
    threshold: float,
) -> pd.DataFrame:
    """Return ``tumor_fraction`` and binary ``label`` for one slide's patches (rows kept in order)."""
    mask, downsample = read_tumor_mask(
        mask_path=mask_path, target_downsample=target_downsample, tumor_value=tumor_value
    )
    level0_coords = slide_patches[["x_level0", "y_level0"]].to_numpy(dtype=np.float64)
    boxes = compute_mask_boxes(level0_coords=level0_coords, footprint=footprint, downsample=downsample)
    fractions = compute_box_fractions(mask=mask, boxes=boxes)
    return pd.DataFrame(
        {"tumor_fraction": fractions, "label": (fractions >= threshold).astype(np.int64)}, index=slide_patches.index
    )


def build_slide_rows(
    *,
    slide_patches: pd.DataFrame,
    slide: str,
    subset: str,
    slide_label: int,
    slide_magnification: float,
    config: Camelyon16MetadataConfig,
) -> pd.DataFrame:
    """Return the metadata rows of one slide: grid coordinates, split, and patch labels."""
    patch_sizes = slide_patches["patch_width"].unique()
    if len(patch_sizes) != 1:
        raise ValueError(f"{slide}: mixed patch sizes {patch_sizes}")
    footprint = compute_patch_footprint(
        patch_size=int(patch_sizes[0]),
        slide_magnification=slide_magnification,
        patch_magnification=config.patch_magnification,
    )
    grid = compute_grid_coordinates(
        level0_coords=slide_patches[["x_level0", "y_level0"]].to_numpy(dtype=np.float64), footprint=footprint
    )
    labels = label_slide_patches(
        slide_patches=slide_patches,
        mask_path=config.mask_dir / f"{slide}{config.mask_suffix}",
        footprint=footprint,
        target_downsample=config.mask_downsample,
        tumor_value=config.tumor_mask_value,
        threshold=config.tumor_fraction_threshold,
    )
    return pd.DataFrame(
        {
            "img_path": slide_patches["patch_path"].to_numpy(),
            "subset": subset,
            "slide": slide,
            "slide_label": slide_label,
            "label": labels["label"].to_numpy(),
            "tumor_fraction": labels["tumor_fraction"].to_numpy(),
            "grid_x": grid[:, 0],
            "grid_y": grid[:, 1],
            "x_level0": slide_patches["x_level0"].to_numpy(),
            "y_level0": slide_patches["y_level0"].to_numpy(),
        }
    )


def load_patch_manifest(pattern: str) -> pd.DataFrame:
    """Concatenate patch manifest shards and add a normalized ``slide`` column.

    Raises:
        FileNotFoundError: If no shard matches.
    """
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No patch manifest matches {pattern}")
    frame = pd.concat([pd.read_csv(path, usecols=MANIFEST_COLUMNS) for path in paths], ignore_index=True)
    frame["slide"] = [normalize_slide_name(path) for path in frame["parent_wsi"]]
    return frame


def load_slide_table(config: Camelyon16MetadataConfig) -> pd.DataFrame:
    """Join slide split/label with slide magnification, indexed by normalized slide name.

    Raises:
        ValueError: If a slide lacks a magnification.
    """
    splits = pd.read_csv(config.split_manifest)
    splits["slide"] = [normalize_slide_name(name) for name in splits["slide_id"]]
    dimensions = pd.read_csv(config.slide_dimensions)
    dimensions["slide"] = [normalize_slide_name(name) for name in dimensions["slide_name"]]
    table = splits.merge(dimensions[["slide", "slide_mag"]], on="slide", how="left").set_index("slide")
    missing = table.index[table["slide_mag"].isna()].tolist()
    if missing:
        raise ValueError(f"No magnification for slides: {missing[:5]}")
    return table


def main() -> None:
    """Write the Camelyon16 patch metadata CSV."""
    """ PARSE ARGS """
    arguments = parse_run_arguments(__doc__ or "")
    config = load_config(Camelyon16MetadataConfig, arguments.config_path, arguments.overrides)
    make_run_directory(arguments.output_dir)
    save_config(config, arguments.output_dir / RESOLVED_CONFIG_FILE)

    """ LOAD MANIFESTS """
    patches = load_patch_manifest(config.patch_manifest_glob)
    slides = load_slide_table(config)
    unknown = sorted(set(patches["slide"]).difference(slides.index))
    if unknown:
        raise ValueError(f"Slides without a split: {unknown[:5]}")
    print(f"{len(patches)} patches from {patches['slide'].nunique()} slides")

    """ LABEL PATCHES """
    slide_info = {
        str(slide): (str(split), int(label), float(magnification))
        for slide, split, label, magnification in zip(
            slides.index, slides["split"], slides["label"], slides["slide_mag"], strict=True
        )
    }
    with ProcessPoolExecutor(max_workers=config.num_workers) as pool:
        futures = [
            pool.submit(
                build_slide_rows,
                slide_patches=slide_patches,
                slide=str(slide),
                subset=slide_info[str(slide)][0],
                slide_label=slide_info[str(slide)][1],
                slide_magnification=slide_info[str(slide)][2],
                config=config,
            )
            for slide, slide_patches in patches.groupby("slide", sort=True)
        ]
        frame = pd.concat([future.result() for future in futures], ignore_index=True)

    normal_with_tumor = frame.loc[(frame["slide_label"] == 0) & (frame["label"] == 1), "slide"].unique()
    if len(normal_with_tumor):
        raise ValueError(f"Normal slides have tumor patches (check tumor_mask_value): {normal_with_tumor[:5]}")

    """ SAVE """
    csv_dir = arguments.output_dir / METADATA_DIRECTORY
    csv_dir.mkdir()
    frame.to_csv(csv_dir / f"{DATASET}{METADATA_SUFFIX}", index=False)
    summary = frame.groupby(["subset", "slide_label"]).agg(slides=("slide", "nunique"), patches=("slide", "size"))
    summary["tumor_patches"] = frame.groupby(["subset", "slide_label"])["label"].sum()
    print(summary.to_string())


if __name__ == "__main__":
    main()
