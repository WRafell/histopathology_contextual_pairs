import numpy as np

from src.config import PairingStrategy
from src.neighbors import (
    MISSING_NEIGHBOR,
    compute_neighbor_table,
    make_chebyshev_offsets,
    select_context_rows,
)


def test_offsets_cover_the_chebyshev_ring() -> None:
    assert len(make_chebyshev_offsets(min_distance=1, max_distance=1)) == 8
    assert len(make_chebyshev_offsets(min_distance=2, max_distance=2)) == 16


def test_neighbor_table_stays_within_slide() -> None:
    # Slide 0: (0,0), (1,0), (5,5). Slide 1: (0,1) would touch (0,0) if slides were mixed.
    slide_ids = np.array([0, 0, 0, 1])
    coords = np.array([[0, 0], [1, 0], [5, 5], [0, 1]])
    table = compute_neighbor_table(slide_ids=slide_ids, coords=coords, min_distance=1, max_distance=1)
    neighbors = [set(row[row != MISSING_NEIGHBOR].tolist()) for row in table]
    assert neighbors == [{1}, {0}, set(), set()]


def test_random_neighbor_falls_back_to_anchor() -> None:
    anchor_rows = np.array([10, 20])
    table = np.array([[3, MISSING_NEIGHBOR], [MISSING_NEIGHBOR, MISSING_NEIGHBOR]])
    rows = select_context_rows(
        strategy=PairingStrategy.RANDOM_NEIGHBOR,
        anchor_rows=anchor_rows,
        neighbor_table=table,
        generator=np.random.default_rng(0),
    )
    assert rows.tolist() == [3, 20]


def test_augmentation_pairs_anchor_with_itself() -> None:
    anchor_rows = np.array([4, 7])
    rows = select_context_rows(
        strategy=PairingStrategy.AUGMENTATION,
        anchor_rows=anchor_rows,
        neighbor_table=None,
        generator=np.random.default_rng(0),
    )
    assert rows.tolist() == [4, 7]
