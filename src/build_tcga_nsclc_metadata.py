"""Build ``tcga_nsclc_data.csv`` from the patch manifests and the patient-level slide split.

Usage:
    python -m src.build_tcga_nsclc_metadata --config configs/tcga_nsclc_metadata.yaml \
        --output-dir runs/metadata/tcga_nsclc_20x_224

Writes ``<output-dir>/csv/tcga_nsclc_data.csv``; use ``<output-dir>`` as ``data.data_root`` with
``data.dataset=tcga_nsclc``. TCGA has no patch annotations, so every ``label`` is -1. Split slides
without patches (no MPP, or no tissue) are listed in ``<output-dir>/slides_without_patches.csv``.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.cli import RESOLVED_CONFIG_FILE, make_run_directory, parse_run_arguments
from src.config import TCGANSCLCMetadataConfig, load_config, save_config
from src.data import METADATA_DIRECTORY, METADATA_SUFFIX
from src.patch_grid import (
    compute_grid_coordinates,
    compute_patch_footprint,
    load_manifest_shards,
    read_base_magnification,
)

DATASET = "tcga_nsclc"
MANIFEST_COLUMNS = ["patch_path", "parent_wsi", "x_level0", "y_level0", "patch_width"]
UNKNOWN_PATCH_LABEL = -1
SLIDES_WITHOUT_PATCHES_FILE = "slides_without_patches.csv"


def build_slide_rows(
    *,
    slide_patches: pd.DataFrame,
    slide: str,
    patient: str,
    subset: str,
    slide_label: int,
    base_magnification: float,
    patch_magnification: float,
) -> pd.DataFrame:
    """Return the metadata rows of one slide: grid coordinates, split, and the unknown patch label.

    Raises:
        ValueError: If the slide mixes patch sizes or its patches are off the grid.
    """
    patch_sizes = slide_patches["patch_width"].unique()
    if len(patch_sizes) != 1:
        raise ValueError(f"{slide}: mixed patch sizes {patch_sizes}")
    footprint = compute_patch_footprint(
        patch_size=int(patch_sizes[0]), slide_magnification=base_magnification, patch_magnification=patch_magnification
    )
    level0_coords = slide_patches[["x_level0", "y_level0"]].to_numpy(dtype=np.float64)
    grid = compute_grid_coordinates(level0_coords=level0_coords, footprint=footprint)
    return pd.DataFrame(
        {
            "img_path": slide_patches["patch_path"].to_numpy(),
            "subset": subset,
            "slide": slide,
            "patient": patient,
            "slide_label": slide_label,
            "label": UNKNOWN_PATCH_LABEL,
            "grid_x": grid[:, 0],
            "grid_y": grid[:, 1],
            "x_level0": slide_patches["x_level0"].to_numpy(),
            "y_level0": slide_patches["y_level0"].to_numpy(),
            "base_magnification": base_magnification,
        }
    )


def build_slide_rows_from_wsi(
    *, slide_patches: pd.DataFrame, slide: str, patient: str, subset: str, slide_label: int, patch_magnification: float
) -> pd.DataFrame:
    """Read the slide's base magnification from its WSI, then build its rows (see ``build_slide_rows``)."""
    return build_slide_rows(
        slide_patches=slide_patches,
        slide=slide,
        patient=patient,
        subset=subset,
        slide_label=slide_label,
        base_magnification=read_base_magnification(Path(slide_patches["parent_wsi"].iloc[0])),
        patch_magnification=patch_magnification,
    )


def load_patch_manifest(pattern: str) -> pd.DataFrame:
    """Concatenate patch manifest shards and add the ``slide`` name (WSI file stem)."""
    frame = load_manifest_shards(pattern, MANIFEST_COLUMNS)
    frame["slide"] = [Path(path).stem for path in frame["parent_wsi"]]
    return frame


def check_subsets_have_all_classes(frame: pd.DataFrame) -> None:
    """Raise ``ValueError`` if a subset lacks a slide class present in the data."""
    classes = set(frame["slide_label"])
    for subset, group in frame.groupby("subset"):
        missing = classes.difference(group["slide_label"])
        if missing:
            raise ValueError(f"Subset {subset} has no slides of class {sorted(missing)}")


def main() -> None:
    """Write the TCGA-NSCLC patch metadata CSV."""
    """ PARSE ARGS """
    arguments = parse_run_arguments(__doc__ or "")
    config = load_config(TCGANSCLCMetadataConfig, arguments.config_path, arguments.overrides)
    make_run_directory(arguments.output_dir)
    save_config(config, arguments.output_dir / RESOLVED_CONFIG_FILE)

    """ LOAD MANIFESTS """
    patches = load_patch_manifest(config.patch_manifest_glob)
    split = pd.read_csv(config.split_file).set_index("slide")
    unknown = sorted(set(patches["slide"]).difference(split.index))
    if unknown:
        raise ValueError(f"Patched slides missing from the split: {unknown[:5]}")
    without_patches = split.loc[~split.index.isin(patches["slide"])]
    without_patches.to_csv(arguments.output_dir / SLIDES_WITHOUT_PATCHES_FILE)
    print(f"{len(patches)} patches from {patches['slide'].nunique()} slides; {len(without_patches)} without patches")

    """ GRID COORDINATES """
    slide_info = {
        str(slide): (str(patient), str(subset), int(label))
        for slide, patient, subset, label in zip(
            split.index, split["patient"], split["subset"], split["slide_label"], strict=True
        )
    }
    with ProcessPoolExecutor(max_workers=config.num_workers) as pool:
        futures = [
            pool.submit(
                build_slide_rows_from_wsi,
                slide_patches=slide_patches,
                slide=str(slide),
                patient=slide_info[str(slide)][0],
                subset=slide_info[str(slide)][1],
                slide_label=slide_info[str(slide)][2],
                patch_magnification=config.patch_magnification,
            )
            for slide, slide_patches in patches.groupby("slide", sort=True)
        ]
        frame = pd.concat([future.result() for future in futures], ignore_index=True)
    check_subsets_have_all_classes(frame)

    """ SAVE """
    csv_dir = arguments.output_dir / METADATA_DIRECTORY
    csv_dir.mkdir()
    frame.to_csv(csv_dir / f"{DATASET}{METADATA_SUFFIX}", index=False)
    summary = frame.groupby(["subset", "slide_label"]).agg(
        patients=("patient", "nunique"), slides=("slide", "nunique"), patches=("slide", "size")
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
