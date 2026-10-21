"""Typed experiment configurations loaded from YAML files with CLI overrides."""

from __future__ import annotations

import dataclasses
import types
import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from pathlib import Path

import yaml

ConfigT = typing.TypeVar("ConfigT")

OVERRIDE_SEPARATOR = "="
KEY_SEPARATOR = "."


class PairingStrategy(StrEnum):
    """How the contextual positive of an anchor patch is chosen."""

    AUGMENTATION = "augmentation"
    RANDOM_NEIGHBOR = "random_neighbor"


class SSLFramework(StrEnum):
    """Supported negative-free self-supervised objectives."""

    VICREG = "vicreg"
    BARLOW_TWINS = "barlow_twins"
    BYOL = "byol"


class Aggregator(StrEnum):
    """Supported slide-level aggregators over frozen patch embeddings."""

    ABMIL = "abmil"
    MEAN_POOLING = "mean_pooling"


@dataclass(frozen=True)
class DataConfig:
    """Location of a dataset's patch metadata and images."""

    dataset: str
    data_root: Path


@dataclass(frozen=True)
class BackboneConfig:
    """Patch encoder architecture."""

    name: str
    is_pretrained: bool
    image_size: int


@dataclass(frozen=True)
class PairingConfig:
    """Positive-pair construction and the contextual loss weight alpha."""

    strategy: PairingStrategy
    min_distance: int
    max_distance: int
    context_weight: float


@dataclass(frozen=True)
class VICRegConfig:
    """VICReg projector and loss coefficients."""

    projector_dims: tuple[int, ...]
    invariance_weight: float
    variance_weight: float
    covariance_weight: float


@dataclass(frozen=True)
class BarlowTwinsConfig:
    """Barlow Twins projector and redundancy-reduction weight."""

    projector_dims: tuple[int, ...]
    off_diagonal_weight: float


@dataclass(frozen=True)
class BYOLConfig:
    """BYOL projector/predictor sizes and target momentum."""

    hidden_dim: int
    projection_dim: int
    target_momentum: float


@dataclass(frozen=True)
class OptimizerConfig:
    """LARS optimizer with linear warm-up and cosine decay."""

    base_learning_rate: float
    weight_decay: float
    warmup_epochs: int


@dataclass(frozen=True)
class SSLTrainConfig:
    """Complete configuration of one SSL pretraining run."""

    seed: int
    device: str
    use_mixed_precision: bool
    data: DataConfig
    backbone: BackboneConfig
    pairing: PairingConfig
    framework: SSLFramework
    optimizer: OptimizerConfig
    batch_size: int
    num_workers: int
    max_epochs: int
    max_anchors: int | None
    initial_checkpoint: Path | None
    vicreg: VICRegConfig
    barlow_twins: BarlowTwinsConfig
    byol: BYOLConfig


@dataclass(frozen=True)
class EmbeddingConfig:
    """Frozen-encoder patch embedding extraction."""

    device: str
    use_mixed_precision: bool
    ssl_run_dir: Path
    batch_size: int
    num_workers: int


@dataclass(frozen=True)
class MILConfig:
    """Slide-level classifier training on frozen patch embeddings."""

    seed: int
    device: str
    embedding_dir: Path
    aggregator: Aggregator
    attention_dim: int
    learning_rate: float
    weight_decay: float
    max_epochs: int


@dataclass(frozen=True)
class LinearProbeConfig:
    """Patch-level linear probe on frozen patch embeddings."""

    seed: int
    device: str
    embedding_dir: Path
    learning_rate: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    ignored_labels: tuple[int, ...] = field(default=())


@dataclass(frozen=True)
class Camelyon16MetadataConfig:
    """Inputs and labeling rule for building the Camelyon16 patch metadata."""

    patch_manifest_glob: str
    split_manifest: Path
    slide_dimensions: Path
    mask_dir: Path
    mask_suffix: str
    patch_magnification: float
    mask_downsample: float
    tumor_mask_value: int
    tumor_fraction_threshold: float
    num_workers: int


def parse_overrides(overrides: Sequence[str]) -> dict[str, object]:
    """Parse ``a.b=value`` strings into a nested mapping; values are YAML-typed.

    Raises:
        ValueError: If an override has no ``=``.
    """
    nested: dict[str, object] = {}
    for override in overrides:
        key, separator, raw_value = override.partition(OVERRIDE_SEPARATOR)
        if not separator:
            raise ValueError(f"Override must look like key=value, got {override!r}")
        *parents, leaf = key.split(KEY_SEPARATOR)
        node = nested
        for parent in parents:
            node = typing.cast(dict[str, object], node.setdefault(parent, {}))
        node[leaf] = yaml.safe_load(raw_value)
    return nested


def merge_mappings(base: Mapping[str, object], update: Mapping[str, object]) -> dict[str, object]:
    """Return ``base`` recursively updated with ``update`` without mutating either."""
    merged = dict(base)
    for key, value in update.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = merge_mappings(typing.cast(Mapping[str, object], current), value)
        else:
            merged[key] = value
    return merged


def build_dataclass(config_type: type[ConfigT], values: Mapping[str, object]) -> ConfigT:
    """Build a (nested) dataclass from a mapping, converting values to the field types.

    Raises:
        ValueError: If keys are unknown or required keys are missing.
    """
    fields = {item.name: item for item in dataclasses.fields(typing.cast(typing.Any, config_type))}
    unknown = set(values).difference(fields)
    if unknown:
        raise ValueError(f"Unknown keys for {config_type.__name__}: {sorted(unknown)}")
    hints = typing.get_type_hints(config_type)
    arguments: dict[str, object] = {}
    for name, item in fields.items():
        if name in values:
            arguments[name] = _convert_value(hints[name], values[name])
        elif item.default is dataclasses.MISSING and item.default_factory is dataclasses.MISSING:
            raise ValueError(f"Missing key {name!r} for {config_type.__name__}")
    return config_type(**arguments)


def _convert_value(hint: object, value: object) -> object:
    """Convert a YAML value to the annotated field type."""
    origin = typing.get_origin(hint)
    if origin in (typing.Union, types.UnionType):
        if value is None:
            return None
        hint = next(arg for arg in typing.get_args(hint) if arg is not type(None))
        origin = typing.get_origin(hint)
    if dataclasses.is_dataclass(hint) and isinstance(value, Mapping):
        return build_dataclass(typing.cast(type, hint), typing.cast(Mapping[str, object], value))
    if origin is tuple:
        return tuple(typing.cast(Sequence[object], value))
    if isinstance(hint, type) and issubclass(hint, (Enum, Path)):
        return hint(typing.cast(str, value))
    return value


def load_config(config_type: type[ConfigT], path: Path, overrides: Sequence[str]) -> ConfigT:
    """Load a YAML configuration file, apply CLI overrides, and build the typed config."""
    with path.open() as stream:
        raw = yaml.safe_load(stream) or {}
    return build_dataclass(config_type, merge_mappings(raw, parse_overrides(overrides)))


def make_plain_mapping(config: object) -> dict[str, object]:
    """Return a YAML-serializable mapping of a dataclass configuration."""
    return typing.cast(dict[str, object], _make_plain_value(dataclasses.asdict(typing.cast(typing.Any, config))))


def _make_plain_value(value: object) -> object:
    """Recursively convert paths, enums, and tuples to YAML-safe values."""
    if isinstance(value, dict):
        return {key: _make_plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_plain_value(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    return value


def save_config(config: object, path: Path) -> None:
    """Write the fully resolved configuration to ``path`` as YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        yaml.safe_dump(make_plain_mapping(config), stream, sort_keys=False)
