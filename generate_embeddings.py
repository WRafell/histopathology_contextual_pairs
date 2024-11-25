from src.utils_common import set_manual_seed, get_basename
from src.utils_data import load_dataset
from src.utils_models import get_backbone
from src.utils_embeddings import save_embeddings
from omegaconf import OmegaConf
from pathlib import Path
import torch
import hydra
import pandas as pd


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args: OmegaConf) -> None:

    set_manual_seed(random_seed=args.random_seed)
    basename = get_basename(args=args)
    print(f"Basename: {basename}")


    """ LOAD DATASET """
    print(f"\nDataset: {args.dataset}")
    df: pd.DataFrame = load_dataset(args.dataset.lower())


    """ LOAD ENCODER """
    print(f"\nSSL Method: {args.ssl_method} with {args.backbone_name} as backbone")     
    encoder, _ = get_backbone(
        backbone_name=args.backbone_name,
        img_size=args.patch_resize)
    
    encoder_path = f"models/{args.dataset.lower()}/{basename}.pth"
    print(f"Encoder path: {encoder_path}")
    assert Path(encoder_path).exists(), f"{encoder_path} does not exist"
    state_dict = torch.load(encoder_path, weights_only=True)
    encoder.load_state_dict(state_dict)


    """ CREATING EMBEDDINGS """
    saving_root = f"embeddings/{args.dataset.lower()}/{basename}"
    Path(saving_root).mkdir(exist_ok=True, parents=True)
    emb_data = save_embeddings(
        saving_root=saving_root,
        df=df,
        encoder=encoder,
        img_size=args.patch_resize,
        device=args.device)
    
    emd_df = pd.DataFrame(emb_data, columns=['subset', 'emb_path', 'label']) 
    emd_df.to_csv(f"csv/{args.dataset}_{Path(basename).stem}.csv", index=False)
    return


if __name__=='__main__':
    main()
