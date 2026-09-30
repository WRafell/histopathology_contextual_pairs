"""Dataset metadata loading and spatial-neighbor preparation."""

from __future__ import annotations

from ast import literal_eval
from pathlib import Path
import random

import numpy as np
import pandas as pd


CSV_DIRECTORY = "csv"
DATASET_FILES = {
    "camelyon16": "camelyon16_data.csv",
    "seedp-stomach": "seedp-stomach_data.csv",
    "seedp-colon": "seedp-colon_data.csv",
    "tcga": "tcga_data.csv",
    "tcga-nsclc": "tcga_data.csv",
}


def _csv_path(data_root: Path, filename: str) -> Path:
    """Return a metadata CSV path under the local data directory."""
    return data_root / CSV_DIRECTORY / filename


def load_dataset(dataset: str, data_root: str | Path = "local") -> pd.DataFrame:
    """Load a dataset metadata CSV and derive Camelyon16 patch coordinates.

    Args:
        dataset: Dataset key from ``DATASET_FILES``.
        data_root: Local data directory containing a ``csv`` subdirectory.

    Returns:
        Dataset metadata as a DataFrame.

    Raises:
        ValueError: If the dataset key is unsupported.
        FileNotFoundError: If its metadata CSV is missing.
    """
    key = dataset.lower()
    try:
        filename = DATASET_FILES[key]
    except KeyError as error:
        raise ValueError(f"Unsupported dataset: {dataset}") from error

    csv_path = _csv_path(Path(data_root), filename)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset metadata not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    if "img_path" in frame:
        root = Path(data_root)
        frame["img_path"] = frame["img_path"].map(lambda value: _resolve_image_path(value, root))
    if key == "camelyon16":
        frame["coords"] = frame["img_path"].map(_camelyon16_coordinates)
    elif "coords" in frame and len(frame):
        frame["coords"] = frame["coords"].map(_parse_coordinates)
    return frame


def _resolve_image_path(image_path: str, data_root: Path) -> str:
    """Resolve metadata image paths relative to the configured local data root."""
    path = Path(image_path)
    if path.is_absolute() or path.parts[:len(data_root.parts)] == data_root.parts:
        return str(path)
    return str(data_root / path)


def _camelyon16_coordinates(image_path: str) -> tuple[int, int]:
    """Read grid coordinates from a Camelyon16 patch filename."""
    x_value, y_value = map(int, Path(image_path).stem.split("_")[:2])
    return x_value // 512, y_value // 512


def _parse_coordinates(value: object) -> tuple[int, int]:
    """Parse a coordinate pair stored as a tuple/list or a CSV string."""
    parsed = literal_eval(value) if isinstance(value, str) else value
    if not isinstance(parsed, (tuple, list)) or len(parsed) != 2:
        raise ValueError(f"Expected a pair of coordinates, got {value!r}")
    return int(parsed[0]), int(parsed[1])


def load_pivot_dataframe(
    dataset: str,
    columns: list[str],
    data_root: str | Path = "local",
) -> pd.DataFrame:
    """Load precomputed anchor and neighbor metadata for SSL training."""
    csv_path = _csv_path(Path(data_root), f"sampled_{dataset.lower()}.csv")
    if not csv_path.is_file():
        raise FileNotFoundError(f"Precomputed neighbor metadata not found: {csv_path}")
    return pd.read_csv(csv_path, usecols=columns)


def get_normal_label(dataset: str) -> int | None:
    """Return the source-code normal-patch label for a supported dataset."""
    return {"camelyon16": 0, "seedp-stomach": 2, "seedp-colon": 2}.get(dataset.lower())


def process_neighbor_data(frame: pd.DataFrame) -> list[tuple[object, ...]]:
    """Parse CSV-encoded neighbor lists into Python values."""
    required = {"pivot_patch", "slide", "coords", "slide_label", "neighbors"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Neighbor metadata missing columns: {sorted(missing)}")
    return [
        (
            str(row.pivot_patch),
            str(row.slide),
            _parse_coordinates(row.coords),
            int(row.slide_label),
            literal_eval(row.neighbors) if isinstance(row.neighbors, str) else row.neighbors,
        )
        for row in frame.itertuples(index=False)
    ]


def preselect_positive_pairs(
    all_train_df: pd.DataFrame,
    num_patches: int,
    proximity_levels: list[int],
    max_num_neighbors: int = 50,
    random_seed: int | None = None,
) -> pd.DataFrame:
    """Sample anchors and list same-slide patches at each Chebyshev distance."""
    if num_patches < 1 or num_patches > len(all_train_df):
        raise ValueError("num_patches must be between 1 and the training row count")
    if max_num_neighbors < 1:
        raise ValueError("max_num_neighbors must be positive")

    rng = random.Random(random_seed)
    sampled = all_train_df.sample(n=num_patches, random_state=random_seed)
    groups = {
        slide: group
        for slide, group in all_train_df.groupby("slide", sort=False)
    }
    columns = ["slide", "coords", "pivot_patch"] + [
        f"neighbor_level_{distance}" for distance in proximity_levels
    ]
    rows: list[list[object]] = []

    for anchor in sampled.itertuples(index=False):
        neighbors_by_distance = groups[anchor.slide]
        coords = np.asarray([_parse_coordinates(item) for item in neighbors_by_distance["coords"]])
        anchor_coords = np.asarray(_parse_coordinates(anchor.coords))
        distances = np.abs(coords - anchor_coords).max(axis=1)
        row: list[object] = [anchor.slide, anchor.coords, anchor.img_path]
        paths = neighbors_by_distance["img_path"].to_numpy()
        for distance in proximity_levels:
            selected = paths[distances == distance].tolist()
            if distance == 0:
                row.append(anchor.img_path)
            elif not selected:
                row.append(anchor.img_path)
            else:
                row.append(rng.sample(selected, min(max_num_neighbors, len(selected))))
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)
