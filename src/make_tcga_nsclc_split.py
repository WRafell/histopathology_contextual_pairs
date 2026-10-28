"""Select TCGA-NSCLC diagnostic slides and split them into train/val/test at patient level.

Usage:
    python -m src.make_tcga_nsclc_split --config configs/tcga_nsclc_split.yaml --output-dir runs/splits/tcga_nsclc

Writes ``<output-dir>/tcga_nsclc_split.csv`` with one row per slide (``slide``, ``patient``, ``project``,
``slide_label``, ``subset``). The committed split is ``configs/splits/tcga_nsclc_split.csv``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.cli import RESOLVED_CONFIG_FILE, make_run_directory, parse_run_arguments
from src.config import TCGANSCLCSplitConfig, load_config, save_config

SPLIT_FILE = "tcga_nsclc_split.csv"
# Slide labels follow this order: LUAD = 0, LUSC = 1.
PROJECTS = ("LUAD", "LUSC")
SLIDE_SUFFIX = ".svs"
BARCODE_PART_COUNT = 6
SAMPLE_TYPE_LENGTH = 2


@dataclass(frozen=True)
class TCGASlideBarcode:
    """Fields of a TCGA slide barcode such as ``TCGA-05-4244-01Z-00-DX1``."""

    patient: str
    sample_type: str
    slide_type: str


def parse_slide_barcode(slide: str) -> TCGASlideBarcode:
    """Parse a slide name ``<barcode>.<uuid>`` into patient, sample type, and slide type.

    Raises:
        ValueError: If the barcode does not have six dash-separated parts.
    """
    parts = slide.split(".")[0].split("-")
    if len(parts) != BARCODE_PART_COUNT:
        raise ValueError(f"Not a TCGA slide barcode: {slide}")
    return TCGASlideBarcode(patient="-".join(parts[:3]), sample_type=parts[3][:SAMPLE_TYPE_LENGTH], slide_type=parts[5])


def select_slides(
    *, filenames: list[str], project: str, slide_label: int, config: TCGANSCLCSplitConfig
) -> pd.DataFrame:
    """Return the slides of one project that match the slide-type prefix and sample-type codes."""
    rows: list[dict[str, object]] = []
    for filename in filenames:
        slide = filename.removesuffix(SLIDE_SUFFIX)
        barcode = parse_slide_barcode(slide)
        if not barcode.slide_type.startswith(config.slide_type_prefix):
            continue
        if barcode.sample_type not in config.sample_type_codes:
            continue
        rows.append({"slide": slide, "patient": barcode.patient, "project": project, "slide_label": slide_label})
    return pd.DataFrame(rows, columns=["slide", "patient", "project", "slide_label"])


def assign_patient_subsets(
    *, patients: pd.DataFrame, val_fraction: float, test_fraction: float, seed: int
) -> pd.Series:
    """Shuffle the patients of each class and cut them into test, val, and train.

    Args:
        patients: One row per patient with ``patient`` and ``slide_label``.
        val_fraction: Share of each class's patients in ``val`` (rounded).
        test_fraction: Share of each class's patients in ``test`` (rounded).
        seed: Seed of the shuffle; classes are shuffled in label order from one generator.

    Returns:
        The subset of every patient, indexed by patient.

    Raises:
        ValueError: If the fractions leave no training patients.
    """
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError(f"Need non-negative fractions summing below 1, got {val_fraction}, {test_fraction}")
    generator = np.random.default_rng(seed)
    subsets: dict[str, str] = {}
    for _, group in patients.sort_values(["slide_label", "patient"]).groupby("slide_label", sort=True):
        shuffled = generator.permutation(group["patient"].to_numpy())
        test_count = round(len(shuffled) * test_fraction)
        val_count = round(len(shuffled) * val_fraction)
        for position, patient in enumerate(shuffled):
            if position < test_count:
                subsets[str(patient)] = "test"
            elif position < test_count + val_count:
                subsets[str(patient)] = "val"
            else:
                subsets[str(patient)] = "train"
    return pd.Series(subsets, name="subset")


def read_manifest_filenames(manifest_path: Path) -> list[str]:
    """Return the ``filename`` column of a GDC download manifest."""
    return pd.read_csv(manifest_path, sep="\t")["filename"].tolist()


def make_split(config: TCGANSCLCSplitConfig) -> pd.DataFrame:
    """Return the selected slides of both projects with their patient-level subsets.

    Raises:
        ValueError: If a patient appears in both projects.
    """
    manifests = (config.luad_manifest, config.lusc_manifest)
    slides = pd.concat(
        [
            select_slides(filenames=read_manifest_filenames(path), project=project, slide_label=label, config=config)
            for label, (project, path) in enumerate(zip(PROJECTS, manifests, strict=True))
        ],
        ignore_index=True,
    )
    patients = slides[["patient", "slide_label"]].drop_duplicates()
    if patients["patient"].duplicated().any():
        raise ValueError(f"Patients in both projects: {patients.loc[patients['patient'].duplicated(), 'patient'][:5]}")
    subsets = assign_patient_subsets(
        patients=patients, val_fraction=config.val_fraction, test_fraction=config.test_fraction, seed=config.seed
    )
    slides["subset"] = slides["patient"].map(subsets)
    return slides.sort_values(["slide_label", "patient", "slide"]).reset_index(drop=True)


def main() -> None:
    """Write the TCGA-NSCLC slide split."""
    """ PARSE ARGS """
    arguments = parse_run_arguments(__doc__ or "")
    config = load_config(TCGANSCLCSplitConfig, arguments.config_path, arguments.overrides)
    make_run_directory(arguments.output_dir)
    save_config(config, arguments.output_dir / RESOLVED_CONFIG_FILE)

    """ SPLIT """
    split = make_split(config)

    """ SAVE """
    split.to_csv(arguments.output_dir / SPLIT_FILE, index=False)
    summary = split.groupby(["subset", "project"]).agg(patients=("patient", "nunique"), slides=("slide", "size"))
    print(summary.to_string())


if __name__ == "__main__":
    main()
