from copy import deepcopy
import torch
import torch.nn.functional as F


class MLPHead(torch.nn.Module):
    def __init__(self, in_channels: int, mlp_hidden_size: int, projector_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_channels, mlp_hidden_size),
            torch.nn.BatchNorm1d(mlp_hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(mlp_hidden_size, projector_dim))

    def forward(self, x):
        return self.net(x)
    

class NetworkWrapper(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, embedding_dim: int, mlp_hidden_size: int, projector_dim: int):
        super().__init__()
        self.encoder = deepcopy(backbone)
        self.projection = MLPHead(embedding_dim, mlp_hidden_size, projector_dim)

    def forward(self, x):
        return self.projection(self.encoder(x))


class BYOL(torch.nn.Module):
    def __init__(self, 
            backbone: torch.nn.Module, 
            embedding_dim: int, 
            mlp_hidden_size: int, 
            projector_dim: int,
            momentum: int):
        super().__init__()
        self.online_network = NetworkWrapper(backbone, embedding_dim, mlp_hidden_size, projector_dim)
        self.target_network = NetworkWrapper(backbone, embedding_dim, mlp_hidden_size, projector_dim)
        self.predictor = MLPHead(
            in_channels=projector_dim, 
            mlp_hidden_size=mlp_hidden_size, 
            projector_dim=projector_dim)
        self.momentum = momentum
        
    @torch.no_grad()
    def update_moving_average(self):
        for online_parameter, target_parameter in zip(self.online_network.parameters(), self.target_network.parameters()):
            target_parameter.data = self.momentum * target_parameter  +  (1. - self.momentum) * online_parameter.data

    def _initialize_target_network(self):
        for online_parameter, target_parameter in zip(self.online_network.parameters(), self.target_network.parameters()):
            target_parameter.data.copy_(online_parameter.data)
            target_parameter.requires_grad = False  # not updating

    @staticmethod
    def loss_function(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def forward(self, view_1, view_2):

        predictions_from_view_1 = self.predictor(self.online_network(view_1))
        predictions_from_view_2 = self.predictor(self.online_network(view_2))

        # compute key features
        with torch.no_grad():
            targets_to_view_2 = self.target_network(view_1)
            targets_to_view_1 = self.target_network(view_2)

        loss = self.loss_function(predictions_from_view_1, targets_to_view_1.detach())
        loss += self.loss_function(predictions_from_view_2, targets_to_view_2.detach())
        return loss.mean()
    
    def get_encoder(self):
        return deepcopy(self.online_network.encoder)

    

        

        