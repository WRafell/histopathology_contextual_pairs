from ast import literal_eval
from typing import Callable
from PIL import Image
import torch
import random
import numpy as np
import lightning as L


class EncoderWithLinear(torch.nn.Module):
    def __init__(self, encoder, embedding_dim, num_classes):
        super(EncoderWithLinear, self).__init__()
        self.encoder = encoder
        self.linear = torch.nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x):
        features = self.encoder(x)
        output = self.linear(features)
        return output

    
# (old) 'img_path', 'patch_label', 'slide_label', 'slide_name', 'coord'
class TwoViewsDataset(torch.utils.data.Dataset):
    def __init__(self, data: np.array, transforms: Callable):
        super().__init__()
        self.img_paths = [x[0] for x in data]
        self.slide_names = [x[1] for x in data]
        self.coords = [x[2] for x in data]           
        if isinstance(self.coords[0], str):
            self.coords = [tuple(map(int, eval(coord))) for coord in self.coords]
        self.neighbors = [literal_eval(x[4]) for x in data] 
        self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        # patch_label = self.patch_labels[idx]
        slide_label = self.slide_labels[idx]
        slide_name = self.slide_names[idx]
        coord = self.coords[idx]
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
    
        view_1, view_2 = self.transforms(img)
        return (view_1, view_2), img_path, slide_name, coord, slide_label


class BasicSSLDataset(torch.utils.data.Dataset):
    def __init__(self, data: np.array, transforms: Callable):
        super().__init__()
        self.img_paths = [x[0] for x in data]
        self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')    
        view_1, view_2 = self.transforms(img)
        return (view_1, view_2), img_path
    

# (new) 'pivot_patch', 'slide', 'coords', 'slide_label', 'neighbors'
class NeighborDataset(torch.utils.data.Dataset):
    def __init__(self, 
            data: list, 
            minimum_distance: int,
            maximum_distance: int,
            transforms: callable,
            resolve_empty_neighbor: bool = True):
        super().__init__()

        assert minimum_distance is not None, "Minimum distance cannot be None"
        assert maximum_distance >= minimum_distance, "Minimum_distance cannot be greater than maximum_distance"

        self.ori_img_paths = [x[0] for x in data]    
        print(f"Image paths before: {len(self.ori_img_paths)}") 
        self.neighbors = [literal_eval(x[1]) for x in data] 
        self.minimum_distance = minimum_distance
        self.maximum_distance = maximum_distance
        self.resolve_empty_neighbor = resolve_empty_neighbor

        # Preprocess neighbors during initialization
        self.processed_neighbors = []
        self.img_paths = []
        for idx, _ in enumerate(self.ori_img_paths):
            neighbors = self._process_neighbors(idx)
            if neighbors is None:
                continue
            self.img_paths.append(self.ori_img_paths[idx])
            self.processed_neighbors.append(neighbors)

        print(f"Image paths after processed: {len(self.img_paths)}")
        print(f"Neighbors: {len(self.processed_neighbors)}")
            
        self.transforms = transforms
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):

        pivot_path = self.img_paths[idx]

        neighbor_path = random.choice(self.processed_neighbors[idx])

        if neighbor_path is None:
            pivot_img = self._load_image(pivot_path)

        pivot_img = self._load_image(pivot_path)
        neighbor_img = self._load_image(neighbor_path)   

        pivot_img, pivot_view = self.transforms(pivot_img)
        
        neighbor_img, _ = self.transforms(neighbor_img)

        return (pivot_img, pivot_view, neighbor_img), (pivot_path, neighbor_path)

    def _load_image(self, img_path: str):
        with open(img_path, 'rb') as f:
            return Image.open(f).convert('RGB')

    def _process_neighbors(self, idx: int) -> list:
        """Preprocess neighbor list for a given index"""
      
        neighbors = self.neighbors[idx] 
        
        # Apply distance cutoff
        if self.maximum_distance is not None:
            neighbors = neighbors[self.minimum_distance:(self.maximum_distance+1)]
        else:
            neighbors = neighbors[self.minimum_distance:]
        
        # Flatten nested structure
        flat_neighbors = [item for sublist in neighbors for item in sublist]

        if flat_neighbors:
            return flat_neighbors
        elif self.resolve_empty_neighbor:       
            # Fallback to pivot path if no neighbors
            return [self.ori_img_paths[idx]]
        else:
            return None
  

class PreselectedPairsDataset(torch.utils.data.Dataset):
    def __init__(self, data, transforms_pivot, transforms_neighbor):
        super().__init__()
        self.pivot_paths = [x[0] for x in data]
        self.neighbor_paths = [x[1] for x in data]
        self.transforms_pivot = transforms_pivot
        self.transforms_neighbor = transforms_neighbor
    
    def __len__(self):
        return len(self.pivot_paths)
    
    def __getitem__(self, idx):
        pivot_path = self.pivot_paths[idx]
        neighbor_path = self.neighbor_paths[idx]
        with open(pivot_path, 'rb') as f:
            pivot_patch = Image.open(f).convert('RGB')
        with open(neighbor_path, 'rb') as f:
            neighbor_patch = Image.open(f).convert('RGB')
        view_1 = self.transforms_pivot(pivot_patch)
        view_2 = self.transforms_neighbor(neighbor_patch)
        return (view_1, view_2), (pivot_path, neighbor_path)
        

class SSLModule(L.LightningModule):
    def __init__(
            self,
            ssl_model: torch.nn.Module,
            positive_pairs_sampling_method: Callable,
            alpha: float,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            saving_path: str,
            use_moving_average: bool):
        super().__init__()
        self.ssl_model = ssl_model
        self.positive_pairs_sampling_method = positive_pairs_sampling_method
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.saving_path = saving_path 
        self.use_moving_average = use_moving_average
        self.temp_step_losses = []
        self.step_losses = []
        self.epoch_losses = []
        self.alpha = alpha

    def training_step(self, batch, batch_idx):
        views = self.positive_pairs_sampling_method(batch)
        pivot_patch = views[0]
        pivot_view = views[1]

        if len(views) == 2:    
            loss = self.ssl_model(pivot_patch, pivot_view)  
        else:            
            another_patch = views[2]
            same_patch_loss = self.ssl_model.forward(pivot_patch, pivot_view) 
            diff_patch_loss = self.ssl_model.forward(pivot_patch, another_patch)
            loss = (1 - self.alpha) * same_patch_loss + self.alpha * diff_patch_loss

        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=views[0].size(0), sync_dist=True)
        self.step_losses.append(loss.item())
        self.temp_step_losses.append(loss.item())
        return loss

    def on_train_epoch_end(self):
        epoch_loss = torch.tensor(self.temp_step_losses).mean()
        self.epoch_losses.append(epoch_loss.item())
        self.temp_step_losses = []

        encoder = self.get_encoder()
        torch.save(encoder.state_dict(), self.saving_path)

    def on_before_zero_grad(self, _):
        if self.use_moving_average:
            self.ssl_model.update_moving_average()

    def configure_optimizers(self):
        if self.scheduler is not None:
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": self.scheduler}
        else:
            return self.optimizer

    def get_encoder(self):
        return self.ssl_model.get_encoder()
    






