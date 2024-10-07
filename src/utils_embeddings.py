from torchvision.transforms import v2
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
import pickle
import pandas as pd


def get_transforms(img_size: int):
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize(size=(img_size, img_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def save_embeddings(saving_root: str, df: pd.DataFrame, encoder: torch.nn.Module, img_size: int, device: str) -> list:

    encoder.eval()
    encoder.to(device)
    transforms = get_transforms(img_size=img_size)
    emb_data = []
    for subset in ['train', 'val', 'test']:
        subdf = df.query(f"subset=='{subset}'")
        slide_names = subdf.slide.unique().tolist()
        for slide_name in tqdm(slide_names, total=len(slide_names), desc=f"{subset}. Embeddings..."):
            slide_df = subdf.query(f"slide=='{slide_name}'")
            slide_label = slide_df.slide_label.unique().tolist()[0]
            saving_path = f"{saving_root}/{subset}/{slide_label}"
            Path(saving_path).mkdir(parents=True, exist_ok=True)
            emb_file = f"{saving_path}/{slide_name}.pkl"
            if Path(emb_file).exists():
                continue        
            img_paths = slide_df.img_path.tolist()
            embeddings = []
            with torch.no_grad():
                for img_path in tqdm(img_paths, total=len(img_paths), leave=False):
                    with open(img_path, 'rb') as f:
                        img = Image.open(f).convert('RGB')
                    img = transforms(img).unsqueeze(0).to(device)
                    embedding = encoder(img).squeeze(0)
                    embeddings.append(embedding.cpu())
            embeddings = torch.stack(embeddings, dim=0)
            
            with open(emb_file, 'wb') as f:
                pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
            emb_data.append([subset, emb_file, slide_label])
    return emb_data


