import numpy as np
import pytest

from src.build_camelyon16_metadata import (
    compute_box_fractions,
    compute_grid_coordinates,
    compute_mask_boxes,
    normalize_slide_name,
)


def test_normalize_slide_name() -> None:
    assert normalize_slide_name("normal-001") == "normal_001"
    assert normalize_slide_name("/x/images/tumor_002.tif") == "tumor_002"


def test_grid_coordinates_from_fractional_footprint() -> None:
    footprint = 224 * 41.13635 / 20
    level0 = np.array([[1382, 2764], [2764, 1382]])
    assert compute_grid_coordinates(level0_coords=level0, footprint=footprint).tolist() == [[3, 6], [6, 3]]
    with pytest.raises(ValueError, match="off the grid"):
        compute_grid_coordinates(level0_coords=np.array([[230.0, 0.0]]), footprint=footprint)


def test_box_fractions() -> None:
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2, :2] = True
    boxes = np.array([[0, 0, 2, 2], [0, 0, 4, 4], [2, 2, 4, 4], [3, 3, 9, 9], [5, 5, 6, 6]])
    assert compute_box_fractions(mask=mask, boxes=boxes).tolist() == [1.0, 0.25, 0.0, 0.0, 0.0]


def test_mask_boxes_cover_footprint() -> None:
    boxes = compute_mask_boxes(level0_coords=np.array([[0.0, 32.0]]), footprint=40.0, downsample=16.0)
    assert boxes.tolist() == [[0, 2, 3, 5]]
