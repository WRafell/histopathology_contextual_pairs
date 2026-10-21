"""Embed every patch of a dataset with the frozen encoder of an SSL run.

Usage:
    python -m src.extract_embeddings --config configs/embeddings.yaml --output-dir runs/embeddings/<name> \
        --set ssl_run_dir=runs/ssl/<name>
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.backbones import make_backbone
from src.cli import RESOLVED_CONFIG_FILE, make_run_directory, parse_run_arguments
from src.config import EmbeddingConfig, SSLTrainConfig, load_config, save_config
from src.data import load_patch_metadata
from src.datasets import ImageDataset
from src.embeddings import MANIFEST_FILE, SLIDE_DIRECTORY, encode_images, make_slide_embeddings, save_slide_embeddings
from src.train_ssl import ENCODER_FILE
from src.transforms import make_eval_transform


def main() -> None:
    """Write per-slide embeddings and a manifest for one SSL run."""
    """ PARSE ARGS """
    arguments = parse_run_arguments(__doc__ or "")
    config = load_config(EmbeddingConfig, arguments.config_path, arguments.overrides)
    ssl_config = load_config(SSLTrainConfig, config.ssl_run_dir / RESOLVED_CONFIG_FILE, ())
    make_run_directory(arguments.output_dir)
    save_config(config, arguments.output_dir / RESOLVED_CONFIG_FILE)
    device = torch.device(config.device)
    use_mixed_precision = config.use_mixed_precision and device.type == "cuda"

    """ LOAD ENCODER """
    encoder, _ = make_backbone(
        name=ssl_config.backbone.name, is_pretrained=False, image_size=ssl_config.backbone.image_size
    )
    encoder.load_state_dict(torch.load(config.ssl_run_dir / ENCODER_FILE, map_location="cpu", weights_only=True))
    encoder.to(device)
    transform = make_eval_transform(image_size=ssl_config.backbone.image_size)

    """ EMBED SLIDES """
    frame = load_patch_metadata(ssl_config.data.dataset, ssl_config.data.data_root)
    manifest_rows: list[dict[str, object]] = []
    for slide_name, slide_frame in tqdm(frame.groupby("slide", sort=True), desc="Slides"):
        loader = DataLoader(
            ImageDataset(image_paths=slide_frame["img_path"].tolist(), transform=transform),
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
        )
        embeddings = encode_images(
            encoder=encoder, loader=loader, device=device, use_mixed_precision=use_mixed_precision
        )
        slide = make_slide_embeddings(slide_frame=slide_frame, embeddings=embeddings)
        relative_path = Path(SLIDE_DIRECTORY) / f"{slide_name}.pt"
        save_slide_embeddings(arguments.output_dir / relative_path, slide)
        manifest_rows.append(
            {
                "slide": slide_name,
                "subset": slide_frame["subset"].iloc[0],
                "slide_label": slide.slide_label,
                "path": str(relative_path),
            }
        )
    pd.DataFrame(manifest_rows).to_csv(arguments.output_dir / MANIFEST_FILE, index=False)


if __name__ == "__main__":
    main()
