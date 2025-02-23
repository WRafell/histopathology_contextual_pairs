from torchmetrics.classification import MulticlassAUROC
from sklearn.metrics import accuracy_score
from torchvision.transforms import v2
from tqdm import tqdm
from copy import deepcopy
from PIL import Image
import torch
import torch.nn.functional as F


class Transforms:
    def __init__(self, resize_int: int = 224):
        self.train_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.RandomResizedCrop(size=(resize_int, resize_int), scale=(0.8, 1.0), interpolation=Image.BICUBIC, antialias=True),
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
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, label = self.dataset[index]
        label = torch.tensor(label, dtype=torch.long)
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')   
        img = self.transforms(img)
        return img, label, img_path
    

class PatchLevelClassifier(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, embedding_dim: int, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.linear = torch.nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x):
        out = self.linear(self.encoder(x))
        return out
   

class PatchLevelTrainer():
    def __init__(self, model: torch.nn.Module, learning_rate: float, max_epochs: int, device: int, patience: int = None):
        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.max_epoch = max_epochs
        self.device = device
        self.patience = patience

    def train(self, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader):

        self.model.to(self.device)

        self.train_losses = []
        self.val_losses = []
        self.train_acc = []
        self.val_acc = []
        lowest_val_loss = float('inf')
        best_model = deepcopy(self.model)
        best_epoch = 0
        patience_counter = 0
        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(1, self.max_epoch+1):
            self.model.train()
            running_loss = 0.
            total_inputs = 0
            corrects = 0
            for batch in tqdm(train_loader, total=len(train_loader), desc=f"{epoch}/{self.max_epoch}. Train", leave=False):
                inputs = batch[0].to(self.device)
                labels = batch[1].to(self.device)              

                self.optimizer.zero_grad()                
                with torch.cuda.amp.autocast():
                    output = self.model(inputs)
                    loss = self.loss_function(output, labels)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                _, pred = torch.max(output, 1)
                running_loss += loss.item()
                total_inputs += labels.size(0)
                corrects += (pred==labels).sum()
            train_loss = running_loss/total_inputs
            train_acc = corrects/total_inputs
            self.train_losses.append(train_loss)
            self.train_acc.append(train_acc)

            self.model.eval()
            total_inputs = 0
            running_loss = 0.
            corrects = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, total=len(val_loader), desc=f"{epoch}/{self.max_epoch}. Val", leave=False):
                    inputs = batch[0].to(self.device)
                    labels = batch[1].to(self.device)   

                    output = self.model(inputs)
                    loss = F.cross_entropy(output, labels)

                    _, pred = torch.max(output, 1)
                    running_loss += loss.item()
                    total_inputs += labels.size(0)
                    corrects += (pred==labels).sum()
            val_loss = running_loss/total_inputs
            val_acc = corrects/total_inputs
            self.val_losses.append(val_loss)
            self.val_acc.append(val_acc)

            if val_loss < lowest_val_loss:
                lowest_val_loss = val_loss
                best_model = deepcopy(self.model)
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch%10==0) or (epoch==1) or (epoch+1==self.max_epoch):
                print(f"{epoch}/{self.max_epoch} Train loss: {train_loss:.4f}. Train acc: {train_acc:.4f} "
                    f"Val loss: {val_loss:.4f}. Val acc: {val_acc:.4f}. Patience: {patience_counter}/{self.patience}")
                
            if self.patience is not None:
                if patience_counter >= self.patience:
                    print(f"Early stopping triggered. Best epoch: {best_epoch} with Val loss: {lowest_val_loss:.4f}")
                    break


        return best_model
    
    def test(self, model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, num_classes: int):
        model.to(self.device)
        model.eval()
        total_inputs = 0
        running_loss = 0.
        corrects = 0
        all_labels = []
        all_preds = []
        probs = []
        with torch.no_grad():
            for batch in tqdm(test_loader, total=len(test_loader), desc="Testing...", leave=False):
                inputs = batch[0].to(self.device)
                labels = batch[1].to(self.device)  


                outputs = model(inputs)
                loss = F.cross_entropy(outputs, labels)


                _, preds = torch.max(outputs, 1)
                running_loss += loss.item()
                total_inputs += labels.size(0)
                corrects += (preds==labels).sum()

                all_labels.extend(labels.detach().tolist())
                all_preds.extend(preds.detach().tolist())
                probs.extend(outputs.detach().tolist())
        test_loss = running_loss/total_inputs
        test_acc = accuracy_score(all_labels, all_preds)
        metric = MulticlassAUROC(num_classes=num_classes, average="macro")
        auc = metric(torch.tensor(probs), torch.tensor(all_labels).long()).item()
        print(f"Test loss: {test_loss:.4f}. Test acc: {test_acc:.4f}. AUROC: {auc:.4f}")
        return test_loss, test_acc, auc
