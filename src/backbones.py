"""Patch encoder construction by name."""

from __future__ import annotations

from torch import nn

RESNET18 = "resnet18"
RESNET18_DIM = 512
VIT_SMALL = "vit_small"
VIT_SMALL_TIMM_NAME = "vit_small_patch16_224.augreg_in21k_ft_in1k"
VIT_SMALL_DIM = 384


def make_backbone(*, name: str, is_pretrained: bool, image_size: int) -> tuple[nn.Module, int]:
    """Build a patch encoder that maps images to a flat embedding.

    Args:
        name: ``resnet18`` or ``vit_small`` (ViT-S/16).
        is_pretrained: Load ImageNet weights (downloaded on first use).
        image_size: Input side length; ViT position embeddings are resized to it.

    Returns:
        The encoder and its embedding dimension.

    Raises:
        ValueError: If the name is unsupported.
    """
    if name == RESNET18:
        from torchvision.models import ResNet18_Weights, resnet18

        model = resnet18(weights=ResNet18_Weights.DEFAULT if is_pretrained else None)
        # torchvision annotates fc as Linear; Identity removes the ImageNet classifier.
        model.fc = nn.Identity()  # pyrefly: ignore[bad-assignment]
        return model, RESNET18_DIM
    if name == VIT_SMALL:
        import timm

        model = timm.create_model(VIT_SMALL_TIMM_NAME, pretrained=is_pretrained, num_classes=0, img_size=image_size)
        return model, VIT_SMALL_DIM
    raise ValueError(f"Unsupported backbone: {name}")
