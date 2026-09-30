from src.utils_common import set_manual_seed, get_basename
from src.utils_data import load_dataset
from src.utils_models import get_backbone
from src.utils_embeddings import save_embeddings
from omegaconf import DictConfig
from pathlib import Path
import torch
import hydra
import pandas as pd


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(args: DictConfig) -> None:

    set_manual_seed(random_seed=args.random_seed)
    basename = get_basename(args=args)
    print(f"Basename: {basename}")


    """ LOAD DATASET """
    print(f"\nDataset: {args.dataset}")
    df: pd.DataFrame = load_dataset(args.dataset.lower(), args.data_root)


    """ LOAD ENCODER """
    if args.ssl_method.lower() == "dino":
        raise NotImplementedError("DINO is not integrated into the active training pipeline")
    encoder, _ = get_backbone(
        backbone_name=str(args.backbone_name),
        img_size=int(args.patch_resize),
    )
    img_size = int(args.patch_resize)


    encoder_path = Path(args.data_root) / "models" / args.dataset.lower() / f"{basename}.pth"
    print(f"Encoder path: {encoder_path}")
    if not encoder_path.is_file():
        raise FileNotFoundError(f"Encoder checkpoint not found: {encoder_path}")
    state_dict = torch.load(encoder_path, map_location="cpu", weights_only=True)
    encoder.load_state_dict(state_dict)


    """ CREATING EMBEDDINGS """
    saving_root = Path(args.data_root) / "embeddings" / args.dataset.lower() / basename
    Path(saving_root).mkdir(exist_ok=True, parents=True)
    emb_data = save_embeddings(
        saving_root=saving_root,
        df=df,
        encoder=encoder,
        img_size=img_size,
        device=args.device)

    emd_df = pd.DataFrame(emb_data, columns=['subset', 'emb_path', 'label'])
    csv_dir = Path(args.data_root) / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    emd_df.to_csv(csv_dir / f"{args.dataset}_{Path(basename).name}.csv", index=False)
    return


if __name__=='__main__':
    main()
