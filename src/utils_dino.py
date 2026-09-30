"""DINO model components retained from the source implementation."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from src.dino import DINOLoss, DINOHead, MultiCropWrapper
from src.utils_vit import vits_dict


def get_dino_components(
    *,
    num_epochs: int,
    img_size: int,
    architecture: str,
    patch_size: int = 16,
    local_crops_number: int = 0,
    drop_path_rate: float = 0.1,
    out_dim: int = 65536,
    norm_last_layer: bool = False,
    warmup_teacher_temp: float = 0.04,
    final_teacher_temp: float = 0.07,
    warmup_teacher_temp_epochs: int = 30,
    pretrained: bool = False,
) -> tuple[nn.Module, nn.Module, DINOLoss]:
    """Build DINO student, teacher, and loss components."""
    model = vits_dict[architecture]
    student_backbone = model(img_size=img_size, patch_size=patch_size, drop_path_rate=drop_path_rate)
    teacher_backbone = model(img_size=img_size, patch_size=patch_size)
    if pretrained:
        weights_url = (
            "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/"
            "pretrained-weights/dino_vit_small_patch16_ep200.torch"
        )
        state_dict = torch.hub.load_state_dict_from_url(weights_url, progress=False)
        student_backbone.load_state_dict(state_dict)
        teacher_backbone.load_state_dict(state_dict)

    student = MultiCropWrapper(
        student_backbone,
        DINOHead(student_backbone.embed_dim, out_dim, norm_last_layer),
    )
    teacher = MultiCropWrapper(teacher_backbone, DINOHead(teacher_backbone.embed_dim, out_dim))
    teacher.load_state_dict(student.state_dict())
    teacher.requires_grad_(False)
    loss = DINOLoss(
        out_dim=out_dim,
        ncrops=local_crops_number + 2,
        warmup_teacher_temp=warmup_teacher_temp,
        final_teacher_temp=final_teacher_temp,
        warmup_teacher_temp_epochs=warmup_teacher_temp_epochs,
        nepochs=num_epochs,
    )
    return student, teacher, loss


def cosine_scheduler(
    base_value: float,
    final_value: float,
    epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int = 0,
    start_warmup_value: float = 0.0,
) -> np.ndarray:
    """Create a linear-warmup then cosine-decay schedule."""
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    if total_steps < 1 or warmup_steps > total_steps:
        raise ValueError("Schedule length must be positive and warmup cannot exceed training")
    warmup = np.linspace(start_warmup_value, base_value, warmup_steps) if warmup_steps else np.array([])
    decay_steps = total_steps - warmup_steps
    progress = np.arange(decay_steps) / max(1, decay_steps)
    decay = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * progress))
    return np.concatenate((warmup, decay))
