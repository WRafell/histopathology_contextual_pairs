"""Pretrain a patch encoder with augmentation and spatial-context positive pairs.

Usage:
    python -m src.train_ssl --config configs/ssl_vicreg_resnet18.yaml --output-dir runs/ssl/<name> [--set key=value]
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.backbones import make_backbone
from src.cli import (
    RESOLVED_CONFIG_FILE,
    append_json_line,
    make_run_directory,
    parse_run_arguments,
    seed_global_generators,
)
from src.config import PairingStrategy, SSLTrainConfig, load_config, save_config
from src.data import load_patch_metadata, make_coordinate_array, select_anchor_rows
from src.datasets import PositivePairDataset
from src.lars import LARS, compute_peak_learning_rate
from src.neighbors import compute_neighbor_table, select_context_rows
from src.ssl_models import make_ssl_model
from src.ssl_training import LearningRateSchedule, train_one_epoch
from src.transforms import make_view_transforms

CHECKPOINT_FILE = "checkpoint.pt"
ENCODER_FILE = "encoder.pt"
METRICS_FILE = "metrics.jsonl"


def main() -> None:
    """Run one SSL pretraining job."""
    """ PARSE ARGS """
    arguments = parse_run_arguments(__doc__ or "")
    config = load_config(SSLTrainConfig, arguments.config_path, arguments.overrides)
    make_run_directory(arguments.output_dir)
    save_config(config, arguments.output_dir / RESOLVED_CONFIG_FILE)
    seed_global_generators(config.seed)
    device = torch.device(config.device)
    use_mixed_precision = config.use_mixed_precision and device.type == "cuda"

    """ LOAD DATA """
    frame = load_patch_metadata(config.data.dataset, config.data.data_root)
    train_frame = frame[frame["subset"] == "train"].reset_index(drop=True)
    anchor_rows = select_anchor_rows(frame=train_frame, max_anchors=config.max_anchors, seed=config.seed)
    has_context_view = config.pairing.context_weight > 0.0
    neighbor_table = None
    if has_context_view and config.pairing.strategy is not PairingStrategy.AUGMENTATION:
        neighbor_table = compute_neighbor_table(
            slide_ids=train_frame["slide_id"].to_numpy(),
            coords=make_coordinate_array(train_frame),
            min_distance=config.pairing.min_distance,
            max_distance=config.pairing.max_distance,
        )[anchor_rows]
    image_paths = train_frame["img_path"].tolist()
    first_transform, second_transform = make_view_transforms(image_size=config.backbone.image_size)
    pairing_generator = np.random.default_rng(config.seed)
    print(f"{len(anchor_rows)} anchors from {train_frame['slide'].nunique()} training slides")

    """ BUILD MODEL """
    backbone, embedding_dim = make_backbone(
        name=config.backbone.name, is_pretrained=config.backbone.is_pretrained, image_size=config.backbone.image_size
    )
    model = make_ssl_model(config=config, backbone=backbone, embedding_dim=embedding_dim)
    if config.initial_checkpoint is not None:
        model.load_state_dict(torch.load(config.initial_checkpoint, map_location="cpu", weights_only=True)["model"])
    model.to(device)
    optimizer = LARS(model.parameters(), weight_decay=config.optimizer.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_mixed_precision)
    steps_per_epoch = len(anchor_rows) // config.batch_size
    if steps_per_epoch == 0:
        raise ValueError(f"{len(anchor_rows)} anchors cannot fill one batch of {config.batch_size}")
    schedule = LearningRateSchedule(
        total_steps=config.max_epochs * steps_per_epoch,
        warmup_steps=config.optimizer.warmup_epochs * steps_per_epoch,
        peak_learning_rate=compute_peak_learning_rate(
            base_learning_rate=config.optimizer.base_learning_rate, batch_size=config.batch_size
        ),
    )

    """ TRAINING """
    step = 0
    for epoch in range(config.max_epochs):
        started_at = time.perf_counter()
        context_rows = None
        if has_context_view:
            context_rows = select_context_rows(
                strategy=config.pairing.strategy,
                anchor_rows=anchor_rows,
                neighbor_table=neighbor_table,
                generator=pairing_generator,
            )
        dataset = PositivePairDataset(
            image_paths=image_paths,
            anchor_rows=anchor_rows,
            context_rows=context_rows,
            first_transform=first_transform,
            second_transform=second_transform,
        )
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
            generator=torch.Generator().manual_seed(config.seed + epoch),
        )
        result = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            scaler=scaler,
            schedule=schedule,
            first_step=step,
            context_weight=config.pairing.context_weight,
            device=device,
            use_mixed_precision=use_mixed_precision,
        )
        step = result.next_step
        torch.save({"model": model.state_dict(), "epoch": epoch + 1}, arguments.output_dir / CHECKPOINT_FILE)
        torch.save(model.backbone.state_dict(), arguments.output_dir / ENCODER_FILE)
        record: dict[str, object] = {
            "epoch": epoch + 1,
            "loss": result.mean_loss,
            "learning_rate": result.last_learning_rate,
            "skipped_fp16_steps": result.num_skipped_steps,
            "self_pair_fraction": None if context_rows is None else float(np.mean(context_rows == anchor_rows)),
            "seconds": time.perf_counter() - started_at,
        }
        append_json_line(arguments.output_dir / METRICS_FILE, record)
        print(record)


if __name__ == "__main__":
    main()
