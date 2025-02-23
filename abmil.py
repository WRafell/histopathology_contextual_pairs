from sklearn.metrics import confusion_matrix, accuracy_score
from torchmetrics.classification import MulticlassAUROC
from copy import deepcopy
from tqdm import tqdm
import torch
import random
import pickle
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        self.M = embedding_dim
        self.L = 128
        self.ATTENTION_BRANCHES = 1

        self.attention = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.M*self.ATTENTION_BRANCHES, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):

        A = self.attention(x)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        M = torch.mm(A, x)
        out = self.classifier(M)
        return out
    

class ABMIL():
    def __init__(self, embedding_dim: int, num_classes: int, learning_rate: float, weight_decay: float, max_epochs: int, device: str, verbose: bool = True):
        self.model = Attention(embedding_dim, num_classes).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max_epochs)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.max_epochs = max_epochs
        self.device = device
        self.num_classes = num_classes
        self.verbose = verbose
        return
    
    def train(self, train_data: list, val_data: list):

        self.train_losses = []
        self.val_losses = []
        self.train_acc = []
        self.val_acc = []
        lowest_val_loss = 1e3
        best_model = deepcopy(self.model)
        best_epoch = 0
        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(1, self.max_epochs+1):
            self.model.train()      
            total_inputs = len(train_data)
            train_data = random.sample(train_data, total_inputs)
            running_loss = 0.
            corrects = 0
            for emb_file, slide_label in tqdm(train_data, total=total_inputs, desc=f"{epoch}/{self.max_epochs}. Train", leave=False):

                with open(emb_file, 'rb') as f:
                    embeddings = pickle.load(f)
                embeddings = embeddings.to(self.device)
                slide_label = torch.tensor(slide_label).long().unsqueeze(0).to(self.device)

                self.optimizer.zero_grad()                
                with torch.cuda.amp.autocast():
                    output = self.model(embeddings)
                    loss = self.loss_function(output, slide_label)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                pred = output.argmax()
                running_loss += loss.item()
                corrects += (pred==slide_label).sum()
            train_loss = running_loss/total_inputs
            train_acc = corrects/total_inputs
            self.train_losses.append(train_loss)
            self.train_acc.append(train_acc)

            self.scheduler.step()

            self.model.eval()
            total_inputs = len(val_data)
            running_loss = 0.
            corrects = 0
            with torch.no_grad():
                for emb_file, slide_label in tqdm(val_data, total=total_inputs, desc=f"{epoch}/{self.max_epochs}. Val", leave=False):
                    with open(emb_file, 'rb') as f:
                        embeddings = pickle.load(f)
                    embeddings = embeddings.to(self.device)
                    slide_label = torch.tensor(slide_label).long().unsqueeze(0).to(self.device)


                    output = self.model(embeddings)
                    loss = F.cross_entropy(output, slide_label)

                    pred = output.argmax()
                    running_loss += loss.item()
                    corrects += (pred==slide_label).sum()
            val_loss = running_loss/total_inputs
            val_acc = corrects/total_inputs
            self.val_losses.append(val_loss)
            self.val_acc.append(val_acc)

            if val_loss < lowest_val_loss:
                lowest_val_loss = val_loss
                best_model = deepcopy(self.model)

            if not self.verbose:
                continue
            if (epoch%10==0) or (epoch==1) or (epoch+1==self.max_epochs):
                print(f"{epoch}/{self.max_epochs} Train loss: {train_loss:.4f}. Train acc: {train_acc:.4f} "
                    f"Val loss: {val_loss:.4f}. Val acc: {val_acc:.4f}")


        return best_model
    

    def test(self, model: torch.nn.Module, test_data: list):
        self.model.eval()
        total_inputs = len(test_data)
        running_loss = 0.
        corrects = 0
        all_labels = []
        all_preds = []
        probs = []
        with torch.no_grad():
            for emb_file, slide_label in tqdm(test_data, total=total_inputs, desc="Testing...", leave=False):
                with open(emb_file, 'rb') as f:
                    embeddings = pickle.load(f)
                embeddings = embeddings.to(self.device)

                all_labels.append(slide_label)
                slide_label = torch.tensor(slide_label).long().unsqueeze(0).to(self.device)
                output = model(embeddings)
                loss = F.cross_entropy(output, slide_label)

                probs.extend(output.detach().tolist())
                pred = output.argmax()
                all_preds.append(pred.item())
                running_loss += loss.item()
                corrects += (pred==slide_label).sum()
        test_loss = running_loss/total_inputs
        test_acc = accuracy_score(all_labels, all_preds)
        metric = MulticlassAUROC(num_classes=self.num_classes, average="macro")
        auc = metric(torch.tensor(probs), torch.tensor(all_labels).long()).item()
        # print(f"Test loss: {test_loss:.4f}. Test acc: {test_acc:.4f}. AUROC: {auc:.4f}")
        return test_loss, test_acc, auc
