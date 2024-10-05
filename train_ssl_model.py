from src.utils_common import set_manual_seed
from src.utils_data import shuffle_train_data
from src.utils_self_supervised import Transforms, TwoViewsDataset, SSLModule
from omegaconf import OmegaConf
from lightning.pytorch.callbacks import RichProgressBar
import torch
import hydra
import lightning as L

torch.set_float32_matmul_precision('high')


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args: OmegaConf) -> None:

    set_manual_seed(random_seed=args.random_seed)


    # LOAD DATASET
    print(f"\nDataset: {args.dataset}")
    if args.dataset.lower() == 'camelyon16':
        from src.utils_data import get_camelyon16
        df = get_camelyon16(patch_size=args.patch_size)
    else:
        raise ValueError(f"{args.dataset} is not a valid dataset")
        
    train_df = df.query("subset=='train'")
    shuffled_train_df = shuffle_train_data(
        train_df=train_df,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        num_slides_per_batch=args.num_slides_per_batch)
    data = shuffled_train_df[['img_path', 'slide', 'coords']].values.tolist()

    transforms = Transforms(resize_int=args.patch_resize).two_views_transform
    dataset = TwoViewsDataset(data=data, transforms=transforms)
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers)


    # POSITIVE PAIR SAMPLING
    print(f"\nPositive pair sampling: {args.positive_pair_sampling}")
    if args.positive_pair_sampling == 'baseline':
        from src.utils_positive_pairs import two_distortion_positive_pairs
        positive_pairs_sampling_method = two_distortion_positive_pairs
    elif args.positive_pair_sampling == 'closest_patch':
        from src.utils_positive_pairs import closest_patch_positive_pairs
        positive_pairs_sampling_method = closest_patch_positive_pairs
    else:
        raise ValueError(f"{args.positive_pair_sampling} is not a valid positive pair sampling method")


    # SSL MODEL
    backbone_name = args.backbone._target_.split('.')[-1]    
    backbone = hydra.utils.instantiate(args.backbone)
    optimizer = hydra.utils.instantiate(args.optimizer, params=backbone.parameters())
    scheduler = hydra.utils.instantiate(args.scheduler, optimizer=optimizer, T_max=args.max_epochs)
    saving_path = f"models/{args.positive_pair_sampling}_{args.ssl_method}_{backbone_name}.pth"

    print(f"\nSSL Method: {args.ssl_method} with {backbone_name} as backbone")     
    if args.ssl_method == 'barlow_twins':
        from src.ssl_methods.barlow_twins import BarlowTwins
        ssl_model = BarlowTwins(
            backbone=backbone,
            projector=args.barlow_twins.projector,
            lambd=args.barlow_twins.lambd)
    else:
        raise ValueError(f"{args.ssl_method} is not a valid ssl method")    

    ssl_module = SSLModule(
        ssl_model=ssl_model,
        positive_pairs_sampling_method=positive_pairs_sampling_method,
        optimizer=optimizer,
        scheduler=scheduler,
        saving_path=saving_path)


    # TRAINER
    print("\nTraining...")
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        precision="bf16-mixed",
        accelerator="gpu",
        devices=[0],
        callbacks=[RichProgressBar(leave=False)],
        limit_val_batches=0)
    trainer.fit(model=ssl_module, train_dataloaders=train_loader)


    return


if __name__=='__main__':
    main()
