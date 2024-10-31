from torchvision.transforms import v2
from typing import Callable
from PIL import Image
import torch
import lightning as L


class Transforms:
    def __init__(self, resize_int: int = 224):
        self.train_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.RandomResizedCrop(size=(resize_int, resize_int), scale=(0.8, 1.0), interpolation=Image.BICUBIC, antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            
            v2.RandomApply(
                [v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], 
                p=0.5
            ),
            v2.RandomApply([v2.RandAugment()], p=0.2),
            # v2.RandomGrayscale(p=0.1),    
            # v2.RandomApply([v2.ElasticTransform(alpha=150.0)], p=0.2),
            # v2.RandomApply([v2.RandomEqualize()], p=0.2),
            # v2.RandomApply([v2.RandomChannelPermutation()], p=0.2),
            v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))], p=0.2),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.train_transforms_prime = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.RandomResizedCrop(size=(resize_int, resize_int), scale=(0.5, 1.0), interpolation=Image.BICUBIC, antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            
            v2.RandomApply(
                [v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], 
                p=0.5
            ),
            v2.RandomApply([v2.RandAugment()], p=0.2),
            v2.RandomGrayscale(p=0.1),    
            v2.RandomApply([v2.RandomChannelPermutation()], p=0.2),
            v2.RandomApply([v2.ElasticTransform(alpha=150.0)], p=0.2),
            v2.RandomApply([v2.RandomEqualize()], p=0.2),
            v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))], p=0.2),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    def two_views_transform(self, x):
        v1 = self.train_transforms(x)
        v2 = self.train_transforms_prime(x)
        return v1, v2

    
# 'img_path', 'patch_label', 'slide_label', 'slide_name', 'coord'
class TwoViewsDataset(torch.utils.data.Dataset):
    def __init__(self, data, transforms):
        super().__init__()
        self.img_paths = [x[0] for x in data]
        # self.patch_labels = [x[1] for x in data]
        # self.slide_labels = [x[2] for x in data]
        self.slide_names = [x[1] for x in data]
        self.coords = [x[2] for x in data]
        self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        # patch_label = self.patch_labels[idx]
        # slide_label = self.slide_labels[idx]
        slide_name = self.slide_names[idx]
        coord = self.coords[idx]
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        y1, y2 = self.transforms(img)

        return (y1, y2), img_path, slide_name, coord
    

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
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            saving_path: str,):
        super().__init__()
        self.ssl_model = ssl_model
        self.positive_pairs_sampling_method = positive_pairs_sampling_method
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.saving_path = saving_path 

    def training_step(self, batch, batch_idx):
        view_1, view_2 = self.positive_pairs_sampling_method(batch)
        loss = self.ssl_model(view_1, view_2)       
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=view_1.size(0), sync_dist=True)
        return loss
    
    def on_train_epoch_end(self):
        encoder = self.get_encoder()
        torch.save(encoder.state_dict(), self.saving_path)

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.scheduler,
        }

    def get_encoder(self):
        return self.ssl_model.get_encoder()