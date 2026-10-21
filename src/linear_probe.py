"""Patch-level linear probe on frozen embeddings.

Usage:
    python -m src.linear_probe --config configs/linear_probe.yaml --output-dir runs/probe/<name> \
        --set embedding_dir=runs/embeddings/<name>
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional

from src.cli import RESOLVED_CONFIG_FILE, make_run_directory, parse_run_arguments, seed_global_generators, write_json
from src.config import LinearProbeConfig, load_config, save_config
from src.data import SUBSETS
from src.embeddings import MISSING_PATCH_LABEL, load_manifest, load_slide_embeddings
from src.metrics import compute_classification_metrics

RESULTS_FILE = "results.json"


def select_labeled_patches(
    *, embeddings: Tensor, labels: Tensor, ignored_labels: Sequence[int]
) -> tuple[Tensor, Tensor]:
    """Keep patches whose label is known and not ignored."""
    excluded = torch.tensor([MISSING_PATCH_LABEL, *ignored_labels])
    is_kept = ~torch.isin(labels, excluded)
    return embeddings[is_kept].float(), labels[is_kept]


def load_patch_split(
    *, manifest: pd.DataFrame, subset: str, ignored_labels: Sequence[int], device: torch.device
) -> tuple[Tensor, Tensor]:
    """Concatenate labeled patch embeddings of every slide in one subset and move them to ``device``.

    Raises:
        ValueError: If the subset has no labeled patches.
    """
    embeddings: list[Tensor] = []
    labels: list[Tensor] = []
    for path in manifest.loc[manifest["subset"] == subset, "path"]:
        slide = load_slide_embeddings(Path(path))
        kept_embeddings, kept_labels = select_labeled_patches(
            embeddings=slide.embeddings, labels=slide.patch_labels, ignored_labels=ignored_labels
        )
        embeddings.append(kept_embeddings)
        labels.append(kept_labels)
    if not labels or sum(len(item) for item in labels) == 0:
        raise ValueError(f"No labeled patches in subset {subset}")
    return torch.cat(embeddings).to(device), torch.cat(labels).to(device)


@torch.no_grad()
def predict_patches(*, model: nn.Module, embeddings: Tensor, labels: Tensor, batch_size: int) -> tuple[Tensor, float]:
    """Return class probabilities and mean cross-entropy over all patches."""
    model.eval()
    logits = torch.cat([model(batch) for batch in embeddings.split(batch_size)])
    return torch.softmax(logits, dim=1), float(functional.cross_entropy(logits, labels))


def train_linear_probe(
    *,
    train: tuple[Tensor, Tensor],
    validation: tuple[Tensor, Tensor],
    num_classes: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    generator: torch.Generator,
) -> nn.Module:
    """Fit a linear classifier with Adam and return the copy with the lowest validation loss."""
    train_embeddings, train_labels = train
    model = nn.Linear(train_embeddings.shape[1], num_classes).to(train_embeddings.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_model, best_loss = copy.deepcopy(model), float("inf")
    for _ in range(max_epochs):
        model.train()
        order = torch.randperm(len(train_labels), generator=generator).to(train_labels.device)
        for batch_rows in order.split(batch_size):
            loss = functional.cross_entropy(model(train_embeddings[batch_rows]), train_labels[batch_rows])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        _, validation_loss = predict_patches(
            model=model, embeddings=validation[0], labels=validation[1], batch_size=batch_size
        )
        if validation_loss < best_loss:
            best_model, best_loss = copy.deepcopy(model), validation_loss
    return best_model


def main() -> None:
    """Run one linear-probe evaluation."""
    """ PARSE ARGS """
    arguments = parse_run_arguments(__doc__ or "")
    config = load_config(LinearProbeConfig, arguments.config_path, arguments.overrides)
    make_run_directory(arguments.output_dir)
    save_config(config, arguments.output_dir / RESOLVED_CONFIG_FILE)
    seed_global_generators(config.seed)
    device = torch.device(config.device)

    """ LOAD DATA """
    manifest = load_manifest(config.embedding_dir)
    train, validation, test = (
        load_patch_split(manifest=manifest, subset=subset, ignored_labels=config.ignored_labels, device=device)
        for subset in SUBSETS
    )
    num_classes = int(train[1].max()) + 1

    """ TRAINING """
    model = train_linear_probe(
        train=train,
        validation=validation,
        num_classes=num_classes,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        batch_size=config.batch_size,
        max_epochs=config.max_epochs,
        generator=torch.Generator().manual_seed(config.seed),
    )

    """ EVALUATION """
    probabilities, test_loss = predict_patches(
        model=model, embeddings=test[0], labels=test[1], batch_size=config.batch_size
    )
    metrics = compute_classification_metrics(labels=test[1].cpu().numpy(), probabilities=probabilities.cpu().numpy())
    results: dict[str, object] = {
        "test_loss": test_loss,
        "num_test_patches": len(test[1]),
        **dataclasses.asdict(metrics),
    }
    write_json(arguments.output_dir / RESULTS_FILE, results)
    print(results)


if __name__ == "__main__":
    main()
