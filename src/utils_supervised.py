from torchmetrics import Accuracy, AUROC
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from PIL import Image
import torch
import lightning as L
import torch.nn.functional as F


class Transforms:
    def __init__(self, resize_int: int = 224):
        self.train_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.RandomResizedCrop(size=(resize_int, resize_int), scale=(0.5, 1.0), interpolation=Image.BICUBIC, antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomApply([v2.RandomChannelPermutation()], p=0.2),
            v2.RandomApply(
                [v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], 
                p=0.5
            ),
            v2.RandomApply([v2.RandAugment()], p=0.2),
            v2.RandomGrayscale(p=0.1),    
            v2.RandomApply([v2.ElasticTransform(alpha=150.0)], p=0.2),
            v2.RandomApply([v2.RandomEqualize()], p=0.2),
            v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))], p=0.2),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.test_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(resize_int, resize_int), interpolation=Image.BICUBIC, antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


class LabeledDataset(torch.utils.data.Dataset):
    """
    A simple dataset class used for image classification tasks.
    
    Args:
        dataset (list): A list of tuples, containing the path to the image and its corresponding label.
        transforms (Compose): A torchvision Compose object, used to apply transformations to the images.
    """

    def __init__(self, dataset: list, transforms: v2.Compose):
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self):
        # Return the number of items in the dataset
        return len(self.dataset)

    def __getitem__(self, index):
        # Get the image path and label from the dataset
        img_path, label = self.dataset[index]
        
        # Convert the label to a tensor
        label = torch.tensor(label, dtype=torch.long)

        # Open the image and convert it to RGB
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')        

        # Apply the transformations to the image
        img = self.transforms(img)

        # Return the image and its label
        return img, label, img_path
    

class PatchDataModule(L.LightningDataModule):
    def __init__(self, 
            dataset: dict, 
            train_transforms: v2.Compose, 
            test_transforms: v2.Compose, 
            batch_size: int, 
            num_workers: int):
        super().__init__()
        self.dataset = dataset
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_data = LabeledDataset(dataset=self.dataset['train'], transforms=self.train_transforms)
        self.val_data = LabeledDataset(dataset=self.dataset['val'], transforms=self.test_transforms)
        self.test_data = LabeledDataset(dataset=self.dataset['test'], transforms=self.test_transforms)
    
    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
    


class SupervisedClassifier(L.LightningModule):
    def __init__(self, backbone: torch.nn.Module, learning_rate: float, max_epochs: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.loss_func = torch.nn.CrossEntropyLoss()

        # Metrics
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.auroc = AUROC(task='multiclass', num_classes=num_classes)
        self.training_step_preds = []
        self.training_step_targets = []

        self.validation_step_preds = []
        self.validation_step_targets = []
        self.validation_step_probs = []

        self.test_step_preds = []
        self.test_step_targets = []
        self.test_step_probs = []

    def forward(self, inputs):
        return self.backbone(inputs)
    
    def training_step(self, batch, batch_idx):
        inputs = batch[0]
        targets = batch[1]
        outputs = self.forward(inputs)
        loss = self.loss_func(outputs, targets)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=targets.size(0))

        _, preds = outputs.data.max(1)
        self.training_step_preds.extend(preds.tolist())
        self.training_step_targets.extend(targets.tolist())
        return loss
    
    def on_train_epoch_end(self):
        all_preds = torch.tensor(self.training_step_preds).float()
        all_targets = torch.tensor(self.training_step_targets).long()
        acc = self.accuracy(all_preds, all_targets)
        self.log("train_acc", acc, prog_bar=True)

        # Need to empty the lists for the next epoch
        self.training_step_preds.clear()
        self.training_step_targets.clear()
    
    def validation_step(self, batch, batch_idx):
        inputs = batch[0]
        targets = batch[1]
        outputs = self.forward(inputs)
        loss = F.cross_entropy(outputs, targets)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=targets.size(0))        

        _, preds = outputs.data.max(1)
        self.validation_step_preds.extend(preds.cpu().tolist())
        self.validation_step_targets.extend(targets.cpu().tolist())
        self.validation_step_probs.extend(outputs.cpu())

    def on_validation_epoch_end(self):
        all_preds = torch.tensor(self.validation_step_preds).float()
        all_targets = torch.tensor(self.validation_step_targets).long()
        all_probs = torch.stack(self.validation_step_probs).float()        
        acc = self.accuracy(all_preds, all_targets).item()
        auc = self.auroc(all_probs, all_targets).item()
        self.log("val_acc", acc, prog_bar=True)
        self.log ("val_auc", auc, prog_bar=True)

        # Need to empty the lists for the next epoch
        self.validation_step_preds.clear()
        self.validation_step_targets.clear()
        self.validation_step_probs.clear()


    def test_step(self, batch, batch_idx):
        inputs = batch[0]
        targets = batch[1]
        outputs = self.forward(inputs)
        # loss = F.cross_entropy(outputs, targets)
        _, preds = outputs.data.max(1)
        self.test_step_preds.extend(preds.tolist())
        self.test_step_targets.extend(targets.tolist())
        self.test_step_probs.extend(outputs.cpu())

    def on_test_epoch_end(self):
        all_preds = torch.tensor(self.test_step_preds).float()
        all_targets = torch.tensor(self.test_step_targets).long()
        all_probs = torch.stack(self.test_step_probs).float()
        acc = self.accuracy(all_preds, all_targets).item()
        auc = self.auroc(all_probs, all_targets).item()
        self.log("test_acc", acc)
        self.log("test_auc", auc)
        print(f"Test accuracy: {acc:.4f}. Test AUC: {auc:.4f}")

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }



