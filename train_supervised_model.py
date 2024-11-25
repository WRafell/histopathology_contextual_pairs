from src.utils_common import set_manual_seed, get_basename
from src.utils_data import load_dataset
from src.utils_supervised import Transforms, PatchDataModule, SupervisedClassifier
from src.utils_models import get_backbone
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
    df = df[df['label'] == df['slide_label']]  # only patches with annotations
    train_df = df.query("subset=='train'")
    val_df = df.query("subset=='val'")
    test_df = df.query("subset=='test'")
    dataset = {}
    dataset['train'] = train_df[['img_path', 'label']].values.tolist()
    dataset['val'] = val_df[['img_path', 'label']].values.tolist()
    dataset['test'] = test_df[['img_path', 'label']].values.tolist()

    train_transforms = Transforms(resize_int=args.patch_resize).train_transforms
    test_transforms = Transforms(resize_int=args.patch_resize).test_transforms

    data_module = PatchDataModule(
        dataset=dataset,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    if args.dataset.lower() in ['seedp-stomach', 'seedp-colon']:
        num_classes = 3
    elif args.dataset.lower() == 'camelyon16':
        num_classes = 2


    """ LOAD ENCODER """
    print(f"\nSSL Method: {args.ssl_method} with {args.backbone_name} as backbone")     
    encoder, embedding_dim = get_backbone(
        backbone_name=args.backbone_name,
        img_size=args.patch_resize)

    if args.linear_probing:    
        from src.utils_self_supervised import EncoderWithLinear
        encoder_path = f"models/{args.dataset.lower()}/{basename}.pth"
        print(f"Encoder path: {encoder_path}")
        assert Path(encoder_path).exists(), f"{encoder_path} does not exist"
        state_dict = torch.load(encoder_path, weights_only=True)
        encoder.load_state_dict(state_dict)
        for param in encoder.parameters():
            param.requires_grad = False

    model = EncoderWithLinear(
        encoder=encoder, 
        embedding_dim=embedding_dim, 
        num_classes=num_classes)


    

    return


if __name__=='__main__':
    main()
