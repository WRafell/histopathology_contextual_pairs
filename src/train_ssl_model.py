"""Pretrain an encoder with native PyTorch."""

from __future__ import annotations

import pickle
from pathlib import Path
import time

import hydra
import torch
from omegaconf import DictConfig

from src.utils_common import get_basename, set_manual_seed
from src.utils_data import load_pivot_dataframe
from src.utils_models import get_backbone
from src.utils_transforms import StandardTransforms


torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(args: DictConfig) -> None:
    """Run one configured single-device SSL pretraining job."""
    if str(args.ssl_method).lower() == "dino":
        raise NotImplementedError("DINO multi-crop training is not integrated")
    if len(args.devices) != 1:
        raise NotImplementedError("The native trainer currently supports one device")

    sampling_method = str(args.positive_pair_sampling)
    if sampling_method not in {"baseline", "closest_patch", "random_neighbor"}:
        raise ValueError(f"Unsupported positive-pair sampling method: {sampling_method}")

    seed = int(args.random_seed)
    set_manual_seed(seed)
    dataset_name = str(args.dataset).lower()
    run_name = get_basename(args)
    data_root = Path(args.data_root)
    checkpoint_path = data_root / "models" / dataset_name / f"{run_name}.pth"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    dataset, pair_sampler = _build_dataset(args, dataset_name)
    framework = str(args.ssl_method).lower()
    backbone, embedding_dim = get_backbone(str(args.backbone_name), img_size=int(args.patch_resize))
    model = _build_ssl_model(args, framework, backbone, embedding_dim).to(args.device)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        persistent_workers=int(args.num_workers) > 0,
        pin_memory=torch.device(args.device).type == "cuda",
    )
    optimizer, scheduler, lars_schedule = _build_optimizer(args, model, len(loader))

    step_losses: list[float] = []
    epoch_losses: list[float] = []
    global_step = 0
    started_at = time.perf_counter()
    for epoch in range(int(args.max_epochs)):
        model.train()
        epoch_loss = 0.0
        for batch in loader:
            if lars_schedule is not None:
                lars_schedule.step(optimizer, global_step)
            views = [view.to(args.device, non_blocking=True) for view in pair_sampler(batch)]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=torch.device(args.device).type,
                dtype=torch.bfloat16,
                enabled=torch.device(args.device).type == "cuda",
            ):
                loss = _calculate_loss(model, views, float(args.alpha))
            loss.backward()
            optimizer.step()
            if framework == "byol":
                model.update_moving_average()

            loss_value = float(loss.detach().item())
            step_losses.append(loss_value)
            epoch_loss += loss_value
            global_step += 1

        mean_epoch_loss = epoch_loss / max(1, len(loader))
        epoch_losses.append(mean_epoch_loss)
        if scheduler is not None:
            scheduler.step()
        torch.save(model.get_encoder().state_dict(), checkpoint_path)
        print(f"epoch {epoch + 1}/{args.max_epochs}: loss={mean_epoch_loss:.6f}")

    losses_dir = data_root / "ssl_losses" / dataset_name
    losses_dir.mkdir(parents=True, exist_ok=True)
    with (losses_dir / f"{run_name}.pkl").open("wb") as loss_file:
        pickle.dump({"step_losses": step_losses, "epoch_losses": epoch_losses}, loss_file)
    elapsed = time.perf_counter() - started_at
    print(f"Training finished in {elapsed:.1f}s; encoder saved to {checkpoint_path}")


def _build_dataset(args: DictConfig, dataset_name: str):
    """Build the selected dataset and its batch pair sampler."""
    from src.utils_positive_pairs import three_views_positive_pairs, two_distortion_positive_pairs
    from src.utils_self_supervised import BasicSSLDataset, NeighborDataset

    transforms = StandardTransforms(resize_int=int(args.patch_resize))
    if args.positive_pair_sampling == "baseline":
        frame = load_pivot_dataframe(dataset_name, ["pivot_patch"], args.data_root)
        return BasicSSLDataset(frame[["pivot_patch"]].to_numpy(), transforms), two_distortion_positive_pairs

    frame = load_pivot_dataframe(dataset_name, ["pivot_patch", "neighbors"], args.data_root)
    maximum_distance = int(args.maximum_distance) if args.positive_pair_sampling == "closest_patch" else None
    dataset = NeighborDataset(
        frame[["pivot_patch", "neighbors"]].to_numpy(),
        minimum_distance=int(args.minimum_distance),
        maximum_distance=maximum_distance,
        transforms=transforms,
    )
    return dataset, three_views_positive_pairs


def _build_ssl_model(
    args: DictConfig,
    framework: str,
    backbone: torch.nn.Module,
    embedding_dim: int,
) -> torch.nn.Module:
    """Construct a supported self-supervised model."""
    if framework == "barlow_twins":
        from src.barlow_twins import BarlowTwins

        return BarlowTwins(backbone, embedding_dim, args.barlow_twins.projector, float(args.barlow_twins.lambd))
    if framework == "byol":
        from src.byol import BYOL

        model = BYOL(
            backbone,
            embedding_dim,
            int(args.byol.mlp_hidden_size),
            int(args.byol.projector_dim),
            float(args.byol.momentum),
        )
        model._initialize_target_network()
        return model
    if framework == "simsiam":
        from src.simsiam import SimSiam

        return SimSiam(backbone, embedding_dim, int(args.simsiam.encoder_dim), int(args.simsiam.pred_dim))
    if framework == "vicreg":
        from src.vicreg import VICReg

        return VICReg(
            backbone,
            embedding_dim,
            args.vicreg.projector_arch,
            float(args.vicreg.sim_coeff),
            float(args.vicreg.std_coeff),
            float(args.vicreg.cov_coeff),
            use_distributed=False,
        )
    raise ValueError(f"Unsupported SSL framework: {framework}")


def _build_optimizer(args: DictConfig, model: torch.nn.Module, num_batches: int):
    """Build the configured optimizer and optional learning-rate schedule."""
    if args.use_lars_optimizer:
        from src.lars import AdjustLearningRate, LARS, exclude_bias_and_norm

        optimizer = LARS(
            model.parameters(),
            lr=float(args.lars.lr),
            weight_decay=float(args.lars.weight_decay),
            weight_decay_filter=exclude_bias_and_norm,
            lars_adaptation_filter=exclude_bias_and_norm,
        )
        schedule = AdjustLearningRate(
            max_epochs=int(args.max_epochs),
            batch_size=int(args.batch_size),
            base_lr=float(args.lars.base_lr),
            num_batches=num_batches,
        )
        return optimizer, None, schedule

    optimizer = hydra.utils.instantiate(args.optimizer, params=model.parameters())
    scheduler = hydra.utils.instantiate(args.scheduler, optimizer=optimizer, T_max=int(args.max_epochs))
    return optimizer, scheduler, None


def _calculate_loss(model: torch.nn.Module, views: list[torch.Tensor], alpha: float) -> torch.Tensor:
    """Combine augmentation and contextual positive-pair losses."""
    if len(views) == 2:
        return model(views[0], views[1])
    if len(views) == 3:
        augmentation_loss = model(views[0], views[1])
        context_loss = model(views[0], views[2])
        return (1.0 - alpha) * augmentation_loss + alpha * context_loss
    raise ValueError(f"Expected two or three views, got {len(views)}")


if __name__ == "__main__":
    main()
