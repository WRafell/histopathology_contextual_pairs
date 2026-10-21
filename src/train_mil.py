"""Train a slide-level MIL classifier on frozen embeddings and report test metrics.

Usage:
    python -m src.train_mil --config configs/mil.yaml --output-dir runs/mil/<name> \
        --set embedding_dir=runs/embeddings/<name> --set aggregator=abmil
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.cli import (
    RESOLVED_CONFIG_FILE,
    append_json_line,
    make_run_directory,
    parse_run_arguments,
    seed_global_generators,
    write_json,
)
from src.config import MILConfig, load_config, save_config
from src.data import SUBSETS
from src.embeddings import load_manifest, load_slide_embeddings
from src.metrics import compute_classification_metrics
from src.mil import Bag, make_mil_model, predict_bags, train_mil_model

HISTORY_FILE = "history.jsonl"
RESULTS_FILE = "results.json"


def load_bags(manifest: pd.DataFrame, subset: str) -> list[Bag]:
    """Load all slides of one subset as bags.

    Raises:
        ValueError: If the subset has no slides.
    """
    rows = manifest[manifest["subset"] == subset]
    if rows.empty:
        raise ValueError(f"No {subset} slides in the embedding manifest")
    return [
        Bag(embeddings=load_slide_embeddings(Path(path)).embeddings, label=int(label))
        for path, label in zip(rows["path"], rows["slide_label"], strict=True)
    ]


def main() -> None:
    """Run one MIL training and evaluation."""
    """ PARSE ARGS """
    arguments = parse_run_arguments(__doc__ or "")
    config = load_config(MILConfig, arguments.config_path, arguments.overrides)
    make_run_directory(arguments.output_dir)
    save_config(config, arguments.output_dir / RESOLVED_CONFIG_FILE)
    seed_global_generators(config.seed)
    device = torch.device(config.device)

    """ LOAD DATA """
    manifest = load_manifest(config.embedding_dir)
    train_bags, validation_bags, test_bags = (load_bags(manifest, subset) for subset in SUBSETS)
    num_classes = int(manifest["slide_label"].max()) + 1
    embedding_dim = int(train_bags[0].embeddings.shape[1])

    """ TRAINING """
    model = make_mil_model(
        aggregator=config.aggregator,
        embedding_dim=embedding_dim,
        attention_dim=config.attention_dim,
        num_classes=num_classes,
    )
    best_model, history = train_mil_model(
        model=model,
        train_bags=train_bags,
        validation_bags=validation_bags,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        max_epochs=config.max_epochs,
        device=device,
        generator=torch.Generator().manual_seed(config.seed),
    )
    for record in history:
        append_json_line(arguments.output_dir / HISTORY_FILE, dict(record))

    """ EVALUATION """
    probabilities, test_loss = predict_bags(model=best_model, bags=test_bags, device=device)
    labels = np.array([bag.label for bag in test_bags])
    metrics = compute_classification_metrics(labels=labels, probabilities=probabilities)
    results: dict[str, object] = {"test_loss": test_loss, **dataclasses.asdict(metrics)}
    write_json(arguments.output_dir / RESULTS_FILE, results)
    print(results)


if __name__ == "__main__":
    main()
