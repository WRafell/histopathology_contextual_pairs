from torch import nn
import torch
import torch.distributed
import torch.nn.functional as F


class VICReg(nn.Module):
    def __init__(self, 
            backbone: nn.Module, 
            embedding_dim: int, 
            num_features: int, 
            projector_arch: str,
            sim_coeff: float,
            std_coeff: float,
            cov_coeff: float):
        self.backbone = backbone
        self.embedding_dim = embedding_dim
        self.num_features = num_features
        self.projector_arc = projector_arch
        self.projector = Projector(
            embedding_dim=embedding_dim, 
            projector_arch=projector_arch)
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def forward(self, view_1, view_2):
        batch_size = view_2.size(0)
        z1 = self.projector(self.backbone(view_1))
        z2 = self.projector(self.backbone(view_2))

        repr_loss = F.mse_loss(z1, z2)

        x = torch.cat(FullGatherLayer.apply(z1), dim=0)
        y = torch.cat(FullGatherLayer.apply(z2), dim=0)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2 

        cov_x = (x.T @ x) / (batch_size - 1)
        cov_y = (y.T @ y) / (batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(self.num_features) 
        cov_loss += off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )
        return loss



class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]
    

def Projector(embedding_dim: int, projector_arch: str) -> nn.Module:
    mlp_spec = f"{embedding_dim}-{projector_arch}"
    layers = []
    f = list(map(int, mlp_spec.split('-')))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)
    

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


