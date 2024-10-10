from src.utils_common import set_manual_seed
from src.utils_embeddings import save_embeddings
from omegaconf import OmegaConf
from pathlib import Path
import torch
import hydra
import torchvision
import pandas as pd


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args: OmegaConf) -> None:

    set_manual_seed(random_seed=args.random_seed)

    basename = f"{args.positive_pair_sampling}_{args.ssl_method}_{args.backbone_name}"

    """ LOAD DATASET """
    print(f"\nDataset: {args.dataset}")
    if args.dataset.lower() == 'camelyon16':
        from src.utils_data import get_camelyon16
        df = get_camelyon16()
    else:
        raise ValueError(f"{args.dataset} is not a valid dataset")


    """ SSL MODEL """
    if args.backbone_name.lower() == 'resnet18':
        encoder = torchvision.models.resnet18(weights='ResNet18_Weights.DEFAULT')
        encoder.fc = torch.nn.Identity()
    else:
        raise ValueError(f"{args.backbone_name} is not a valid backbone model")
    
    encoder_path = f"models/{basename}.pth"
    assert Path(encoder_path).exists(), f"{encoder_path} does not exist"
    state_dict = torch.load(encoder_path, weights_only=True)
    encoder.load_state_dict(state_dict)


    """ CREATING EMBEDDINGS """
    saving_root = f"embeddings/{args.dataset}/{basename}"
    Path(saving_root).mkdir(exist_ok=True, parents=True)
    emb_data = save_embeddings(
        saving_root=saving_root,
        df=df,
        encoder=encoder,
        img_size=args.patch_resize,
        device=args.device)
    
    emd_df = pd.DataFrame(emb_data, columns=['subset', 'emb_path', 'label']) 
    emd_df.to_csv(f"csv/{args.dataset}_{basename}.csv", index=False)
    return


if __name__=='__main__':
    main()
