from pathlib import Path

import pytest

from src.config import Aggregator, MILConfig, build_dataclass, merge_mappings, parse_overrides


def test_overrides_are_nested_and_typed() -> None:
    assert parse_overrides(["a.b=1", "a.c=x", "d=null"]) == {"a": {"b": 1, "c": "x"}, "d": None}


def test_merge_keeps_untouched_keys() -> None:
    assert merge_mappings({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}) == {"a": {"b": 3, "c": 2}}


def test_build_dataclass_converts_and_rejects_unknown_keys() -> None:
    values = {
        "seed": 0,
        "device": "cpu",
        "embedding_dir": "runs/x",
        "aggregator": "mean_pooling",
        "attention_dim": 8,
        "learning_rate": 0.1,
        "weight_decay": 0.0,
        "max_epochs": 1,
    }
    config = build_dataclass(MILConfig, values)
    assert config.embedding_dir == Path("runs/x")
    assert config.aggregator is Aggregator.MEAN_POOLING
    with pytest.raises(ValueError, match="Unknown keys"):
        build_dataclass(MILConfig, {**values, "typo": 1})
