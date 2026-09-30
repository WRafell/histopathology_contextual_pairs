from timm.models.vision_transformer import VisionTransformer
from functools import partial
from torch import nn


def vit_tiny(img_size, patch_size=16, **kwargs):
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        num_classes=0,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(img_size, patch_size=16, **kwargs):
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        num_classes=0,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(img_size, patch_size=16, **kwargs):
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        num_classes=0,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


vits_dict = {
    'tiny': vit_tiny,
    'small': vit_small,
    'base': vit_base}