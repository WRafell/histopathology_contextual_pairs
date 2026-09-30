from torchvision import models
from torch import nn


def get_backbone(backbone_name: str, img_size: int = None):
    if backbone_name.lower() == 'resnet18':
        backbone = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        backbone.fc = nn.Identity()
        embedding_dim = 512
    elif backbone_name.lower() == 'vit_small':
        from src.utils_vit import vit_small
        assert img_size is not None, "Set image set for vit"
        backbone = vit_small(img_size=img_size)
        embedding_dim = 384
    elif backbone_name.lower() == 'vit_tiny':
        from src.utils_vit import vit_tiny
        assert img_size is not None, "Set image set for vit"
        backbone = vit_tiny(img_size=img_size)
        embedding_dim = 192
    else:
        raise ValueError(f"{backbone_name} is not a valid backbone model")

    return backbone, embedding_dim