"""Spatial neighbor lookup and contextual positive selection on the patch grid."""

from __future__ import annotations

import numpy as np

from src.config import PairingStrategy

MISSING_NEIGHBOR = -1


def make_chebyshev_offsets(*, min_distance: int, max_distance: int) -> np.ndarray:
    """Return all integer ``(dx, dy)`` offsets with Chebyshev norm in ``[min_distance, max_distance]``.

    Raises:
        ValueError: If ``min_distance < 1`` or ``max_distance < min_distance``.
    """
    if min_distance < 1 or max_distance < min_distance:
        raise ValueError(f"Need 1 <= min_distance <= max_distance, got {min_distance}, {max_distance}")
    steps = np.arange(-max_distance, max_distance + 1)
    grid = np.stack(np.meshgrid(steps, steps, indexing="ij"), axis=-1).reshape(-1, 2)
    norms = np.abs(grid).max(axis=1)
    return grid[(norms >= min_distance) & (norms <= max_distance)]


def compute_neighbor_table(
    *, slide_ids: np.ndarray, coords: np.ndarray, min_distance: int, max_distance: int
) -> np.ndarray:
    """Find same-slide patches at every Chebyshev offset of each patch.

    Args:
        slide_ids: ``(N,)`` integer slide codes.
        coords: ``(N, 2)`` non-negative integer grid coordinates; unique within a slide.
        min_distance: Smallest Chebyshev distance of a neighbor.
        max_distance: Largest Chebyshev distance of a neighbor.

    Returns:
        ``(N, K)`` int64 array; entry ``[i, k]`` is the row of the patch at offset ``k`` from
        patch ``i`` in the same slide, or ``MISSING_NEIGHBOR``.

    Raises:
        ValueError: If shapes disagree or a slide has duplicate coordinates.
    """
    if coords.shape != (len(slide_ids), 2):
        raise ValueError(f"coords must have shape ({len(slide_ids)}, 2), got {coords.shape}")
    offsets = make_chebyshev_offsets(min_distance=min_distance, max_distance=max_distance)
    shifted = coords.astype(np.int64) + max_distance
    span = int(shifted.max(initial=0)) + max_distance + 1
    slides = slide_ids.astype(np.int64)
    keys = (slides * span + shifted[:, 0]) * span + shifted[:, 1]
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    if np.any(sorted_keys[1:] == sorted_keys[:-1]):
        raise ValueError("Duplicate coordinates within a slide")

    # ponytail: dense (N, K) table; fine for small max_distance, switch to CSR if K grows large.
    table = np.full((len(keys), len(offsets)), MISSING_NEIGHBOR, dtype=np.int64)
    for column, (dx, dy) in enumerate(offsets):
        query = (slides * span + shifted[:, 0] + dx) * span + shifted[:, 1] + dy
        positions = np.minimum(np.searchsorted(sorted_keys, query), len(keys) - 1)
        is_found = sorted_keys[positions] == query
        table[is_found, column] = order[positions[is_found]]
    return table


def sample_random_neighbors(*, neighbor_table: np.ndarray, generator: np.random.Generator) -> np.ndarray:
    """Pick one existing neighbor per row uniformly at random.

    Returns:
        ``(N,)`` int64 array of selected patch rows, ``MISSING_NEIGHBOR`` for rows without any.
    """
    is_valid = neighbor_table != MISSING_NEIGHBOR
    scores = np.where(is_valid, generator.random(neighbor_table.shape), -1.0)
    columns = scores.argmax(axis=1)
    return neighbor_table[np.arange(len(neighbor_table)), columns]


def select_context_rows(
    *,
    strategy: PairingStrategy,
    anchor_rows: np.ndarray,
    neighbor_table: np.ndarray | None,
    generator: np.random.Generator,
) -> np.ndarray:
    """Return the contextual positive row of every anchor for one epoch.

    ``AUGMENTATION`` pairs each anchor with itself (a third augmented view), so every strategy
    uses the same number of views per step. ``RANDOM_NEIGHBOR`` draws a uniform neighbor from
    ``neighbor_table``, whose rows align with ``anchor_rows`` and hold patch rows; anchors without
    neighbors fall back to themselves.

    Raises:
        ValueError: If a neighbor strategy gets no table, or the table does not align.
    """
    if strategy is PairingStrategy.AUGMENTATION:
        return anchor_rows.copy()
    if neighbor_table is None or len(neighbor_table) != len(anchor_rows):
        raise ValueError("Neighbor strategies need a neighbor table aligned with anchor_rows")
    if strategy is PairingStrategy.RANDOM_NEIGHBOR:
        sampled = sample_random_neighbors(neighbor_table=neighbor_table, generator=generator)
        return np.where(sampled == MISSING_NEIGHBOR, anchor_rows, sampled)
    raise ValueError(f"Unsupported pairing strategy: {strategy}")
