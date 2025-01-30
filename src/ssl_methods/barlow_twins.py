from copy import deepcopy
import torch


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, embedding_dim: int, projector: str, lambd: float):
        super().__init__()
        self.projector = projector
        self.lambd = lambd
        self.backbone = backbone
        sizes = [embedding_dim] + list(map(int, self.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(torch.nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(torch.nn.BatchNorm1d(sizes[i + 1]))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = torch.nn.Sequential(*layers)
        # normalization for the last layer
        self.bn = torch.nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, views, alpha: float = 0.5):

        """
        Modified to accept 2 losses
        """

        view_1 = views[0]
        view_2 = views[1]  

        batch_size = view_1.size(0)
        representations_1 = self.projector(self.backbone(view_1))
        representations_2 = self.projector(self.backbone(view_2))

        same_patch_loss = self._calculate_loss(representations_1, representations_2, batch_size)

        if len(views) == 3:
            another_patch_view = views[2]
            representations_3 = self.projector(self.backbone(another_patch_view)) 
            diff_patch_loss = self._calculate_loss(representations_1, representations_3, batch_size)   
            return (1 - alpha) * same_patch_loss + alpha * diff_patch_loss
        
        return same_patch_loss
    
    def get_encoder(self):
        return deepcopy(self.backbone)
    
    
    def _calculate_loss(self, z1, z2, batch_size):
        c = self.bn(z1).T @ self.bn(z2)
        c.div_(batch_size)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        return on_diag + self.lambd * off_diag

    