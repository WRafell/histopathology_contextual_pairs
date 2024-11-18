from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
import pickle
import pandas as pd


class ImageDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        return self.transform(img)


def get_transforms(img_size: int):
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize(size=(img_size, img_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])



@torch.no_grad()
def save_embeddings(
        saving_root: str, 
        df: pd.DataFrame, 
        encoder: torch.nn.Module, 
        img_size: int, 
        device: str, 
        rewrite: bool = True,
        batch_size: int = 128, 
        num_workers: int = 16) -> list:
    encoder.eval()
    encoder.to(device)
    transforms = get_transforms(img_size=img_size)
    emb_data = []
    for subset in ['train', 'val', 'test']:
        subdf = df[df['subset']==subset]
        for slide_name, slide_df in tqdm(subdf.groupby('slide'), desc=f"{subset}. Embeddings..."):
            slide_label = slide_df['slide_label'].iloc[0]
            saving_path = f"{saving_root}/{subset}/{slide_label}"
            Path(saving_path).mkdir(parents=True, exist_ok=True)
            emb_file = f"{saving_path}/{slide_name}.pkl"
            if Path(emb_file).exists() and (not rewrite):
                emb_data.append([subset, str(emb_file), slide_label])
                continue
            dataset = ImageDataset(slide_df['img_path'].tolist(), transforms)
            dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

            embeddings = []
            for patches in dataloader:
                patches = patches.to(device, non_blocking=True)
                embedding = encoder(patches)
                embeddings.append(embedding.cpu())
            embeddings = torch.cat(embeddings, dim=0)
            with open(emb_file, 'wb') as f:
                pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
            emb_data.append([subset, emb_file, slide_label])
    return emb_data
