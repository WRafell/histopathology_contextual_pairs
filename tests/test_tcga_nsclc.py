from pathlib import Path

import pandas as pd
import pytest

from src.build_tcga_nsclc_metadata import UNKNOWN_PATCH_LABEL, build_slide_rows, check_subsets_have_all_classes
from src.config import TCGANSCLCSplitConfig
from src.make_tcga_nsclc_split import assign_patient_subsets, parse_slide_barcode, select_slides

CONFIG = TCGANSCLCSplitConfig(
    luad_manifest=Path("unused.tsv"),
    lusc_manifest=Path("unused.tsv"),
    slide_type_prefix="DX",
    sample_type_codes=("01",),
    val_fraction=0.1,
    test_fraction=0.2,
    seed=0,
)


def test_parse_slide_barcode() -> None:
    barcode = parse_slide_barcode("TCGA-05-4244-01Z-00-DX1.d4ff32cd-38cf-40ea-8213-45c2b100ac01")
    assert (barcode.patient, barcode.sample_type, barcode.slide_type) == ("TCGA-05-4244", "01", "DX1")
    with pytest.raises(ValueError, match="barcode"):
        parse_slide_barcode("normal_001")


def test_select_slides_keeps_primary_diagnostic_slides() -> None:
    filenames = [
        "TCGA-05-4244-01Z-00-DX1.aaa.svs",
        "TCGA-05-4244-01A-01-TS1.bbb.svs",
        "TCGA-05-4245-02Z-00-DX1.ccc.svs",
    ]
    slides = select_slides(filenames=filenames, project="LUAD", slide_label=0, config=CONFIG)
    assert slides["slide"].tolist() == ["TCGA-05-4244-01Z-00-DX1.aaa"]


def test_patient_subsets_are_stratified_and_seeded() -> None:
    patients = pd.DataFrame({"patient": [f"P{index:02d}" for index in range(40)], "slide_label": [0] * 20 + [1] * 20})
    subsets = assign_patient_subsets(patients=patients, val_fraction=0.1, test_fraction=0.2, seed=0)
    counts = pd.crosstab(patients.set_index("patient")["slide_label"], subsets).to_dict()
    assert counts == {"test": {0: 4, 1: 4}, "train": {0: 14, 1: 14}, "val": {0: 2, 1: 2}}
    again = assign_patient_subsets(patients=patients, val_fraction=0.1, test_fraction=0.2, seed=0)
    assert subsets.equals(again)
    with pytest.raises(ValueError, match="fractions"):
        assign_patient_subsets(patients=patients, val_fraction=0.5, test_fraction=0.5, seed=0)


def test_build_slide_rows_on_40x_grid() -> None:
    patches = pd.DataFrame(
        {"patch_path": ["a.png", "b.png"], "x_level0": [0, 448], "y_level0": [896, 0], "patch_width": [224, 224]}
    )
    rows = build_slide_rows(
        slide_patches=patches,
        slide="S",
        patient="P",
        subset="train",
        slide_label=1,
        base_magnification=40.0,
        patch_magnification=20.0,
    )
    assert rows[["grid_x", "grid_y"]].to_numpy().tolist() == [[0, 2], [1, 0]]
    assert set(rows["label"]) == {UNKNOWN_PATCH_LABEL}


def test_subsets_must_hold_every_class() -> None:
    frame = pd.DataFrame({"subset": ["train", "train", "val"], "slide_label": [0, 1, 0]})
    with pytest.raises(ValueError, match="Subset val"):
        check_subsets_have_all_classes(frame)
