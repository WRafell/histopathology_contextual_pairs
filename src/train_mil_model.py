"""Train and evaluate a slide-level MIL classifier on frozen features."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig

from src.abmil import ABMIL
from src.utils_common import get_basename, set_manual_seed
from src.utils_models import get_backbone


torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(args: DictConfig) -> None:
    """Train one configured MIL run and save its test metrics."""
    set_manual_seed(int(args.random_seed))
    data_root = Path(args.data_root)
    basename = get_basename(args)
    manifest = data_root / "csv" / f"{args.dataset}_{Path(basename).name}.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Embedding manifest not found: {manifest}")

    frame = pd.read_csv(manifest)
    train_data, validation_data, test_data = _split_embedding_rows(frame)
    embedding_dim = _get_embedding_dim(args)
    num_classes = {
        "camelyon16": 2,
        "tcga": 2,
        "tcga-nsclc": 2,
        "seedp-stomach": 3,
        "seedp-colon": 3,
    }.get(str(args.dataset).lower())
    if num_classes is None:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    if str(args.mil.mil_method).lower() != "abmil":
        raise ValueError(f"Unsupported MIL aggregator: {args.mil.mil_method}")
    trainer = ABMIL(
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        learning_rate=float(args.mil.learning_rate),
        weight_decay=float(args.mil.weight_decay),
        max_epochs=int(args.mil.max_epochs),
        device=str(args.device),
    )
    best_model = trainer.train(train_data, validation_data)
    test_loss, test_accuracy, test_auroc = trainer.test(best_model, test_data)

    results_path = data_root / "results" / "mil" / args.dataset
    results_path.mkdir(parents=True, exist_ok=True)
    metrics = {
        "dataset": str(args.dataset),
        "method": str(args.positive_pair_sampling),
        "seed": int(args.random_seed),
        "aggregator": str(args.mil.mil_method),
        "test_loss": float(test_loss),
        "accuracy": float(test_accuracy),
        "auroc": float(test_auroc),
    }
    result_file = results_path / f"{basename.replace('/', '_')}_seed-{args.random_seed}.json"
    with result_file.open("x") as result_stream:
        json.dump(metrics, result_stream, indent=2)
        result_stream.write("\n")
    print(json.dumps(metrics, indent=2))


def _split_embedding_rows(
    frame: pd.DataFrame,
) -> tuple[list[list[object]], list[list[object]], list[list[object]]]:
    """Validate an embedding manifest and return its three data splits."""
    required = {"subset", "emb_path", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Embedding manifest is missing columns: {sorted(missing)}")
    split_data = {
        subset: frame.loc[frame["subset"] == subset, ["emb_path", "label"]].values.tolist()
        for subset in ("train", "val", "test")
    }
    for name, rows in split_data.items():
        if not rows:
            raise ValueError(f"Embedding manifest has no {name} rows")
    return split_data["train"], split_data["val"], split_data["test"]


def _get_embedding_dim(args: DictConfig) -> int:
    """Get the configured encoder output dimension."""
    if str(args.ssl_method).lower() == "dino":
        raise NotImplementedError("DINO is not integrated into the active training pipeline")
    _, embedding_dim = get_backbone(
        backbone_name=str(args.backbone_name),
        img_size=int(args.patch_resize),
    )
    return embedding_dim


if __name__ == "__main__":
    main()
