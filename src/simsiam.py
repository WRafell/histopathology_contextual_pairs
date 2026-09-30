from copy import deepcopy
from torch import nn
import torch


""" https://github.com/facebookresearch/simsiam/blob/main/simsiam/builder.py """
class SimSiamEncoder(nn.Module):
    def __init__(self, backbone: torch.nn.Module, embedding_dim: int, encoder_dim: int):
        super().__init__()
        self.backbone = deepcopy(backbone)
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, encoder_dim, bias=False),
            # nn.BatchNorm1d(encoder_dim, affine=False)
        )
    def forward(self, x):
        return self.projection(self.backbone(x))


class SimSiam(nn.Module):
    def __init__(self, backbone: torch.nn.Module, embedding_dim: int, encoder_dim: int, pred_dim: int):
        super().__init__()

        self.encoder = SimSiamEncoder(
            backbone=backbone,
            embedding_dim=embedding_dim,
            encoder_dim=encoder_dim)

        self.predictor = nn.Sequential(
            nn.Linear(encoder_dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, encoder_dim)
        )

        self.loss_func = nn.CosineSimilarity(dim=1)

    def forward(self, view_1: torch.tensor, view_2: torch.tensor):

        z1 = self.encoder(view_1)
        z2 = self.encoder(view_2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        loss = self.loss_func(p1, z2.detach()).mean()
        loss += self.loss_func(p2, z1.detach()).mean()
        loss *= -0.5
        return loss

    def get_encoder(self):
        return deepcopy(self.encoder.backbone)


