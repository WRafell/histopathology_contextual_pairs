import numpy as np
import pytest

from src.patch_grid import compute_grid_coordinates, get_base_magnification


def test_grid_coordinates_from_fractional_footprint() -> None:
    footprint = 224 * 41.13635 / 20
    level0 = np.array([[1382, 2764], [2764, 1382]])
    assert compute_grid_coordinates(level0_coords=level0, footprint=footprint).tolist() == [[3, 6], [6, 3]]
    with pytest.raises(ValueError, match="off the grid"):
        compute_grid_coordinates(level0_coords=np.array([[230.0, 0.0]]), footprint=footprint)


def test_grid_matches_patcher_origins() -> None:
    # The patcher places patch (column, row) at round(index * stride * base / 20) in level-0 pixels.
    base_magnification = 10.0 / 0.2527
    step = 224 * base_magnification / 20
    indices = np.arange(0, 400, 7)
    level0 = np.stack([np.rint(indices * step), np.rint(indices[::-1] * step)], axis=1)
    grid = compute_grid_coordinates(level0_coords=level0, footprint=step)
    assert grid.tolist() == np.stack([indices, indices[::-1]], axis=1).tolist()


def test_base_magnification_prefers_mpp() -> None:
    assert get_base_magnification({"openslide.mpp-x": "0.25", "openslide.objective-power": "20"}) == 40.0
    assert get_base_magnification({"openslide.objective-power": "20"}) == 20.0
    with pytest.raises(ValueError, match="neither"):
        get_base_magnification({})
