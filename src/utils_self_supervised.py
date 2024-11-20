from typing import Callable
from PIL import Image
import torch
import lightning as L

    
# 'img_path', 'patch_label', 'slide_label', 'slide_name', 'coord'
class TwoViewsDataset(torch.utils.data.Dataset):
    def __init__(self, data, transforms):
        super().__init__()
        self.img_paths = [x[0] for x in data]
        self.slide_names = [x[1] for x in data]
        self.coords = [x[2] for x in data]           
        if isinstance(self.coords[0], str):
            self.coords = [tuple(map(int, eval(coord))) for coord in self.coords]
        self.slide_labels = [x[3] for x in data]  
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
        y1, y2 = self.transforms(img)

        return (y1, y2), img_path, slide_name, coord, slide_label
    

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


    def training_step(self, batch, batch_idx):
        view_1, view_2 = self.positive_pairs_sampling_method(batch)
        loss = self.ssl_model(view_1, view_2)       
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=view_1.size(0), sync_dist=True)
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
    

# class DINOSSLModule(L.LightningModule):
#     '''
#     Mostly copy-paste from:
#     https://github.com/kwea123/dino_pl/blob/master/train.py
#     '''
#     def __init__(
#             self, 
#             student,
#             teacher,
#             loss_function):
#         super().__init__()

#         self.automatic_optimization = False
#         self.student = student
#         self.teacher = teacher
#         self.loss_function = loss_function

#         self.teacher.load_state_dict(self.student.state_dict())
#         for p in self.teacher.parameters():
#             p.requires_grad = False

        







