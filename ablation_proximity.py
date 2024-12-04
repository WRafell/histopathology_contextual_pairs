from src.utils_common import set_manual_seed, get_basename
from src.utils_data import load_preselected_samples
from src.utils_models import get_backbone
from src.utils_positive_pairs import two_distortion_positive_pairs    
from src.utils_transforms import ContextSamplingTransforms 
from src.utils_self_supervised import PreselectedPairsDataset, SSLModule 
from lightning.pytorch.callbacks import RichProgressBar
from omegaconf import OmegaConf
from pathlib import Path
import time
import torch
import hydra
import pickle
import numpy as np
import lightning as L


torch.set_float32_matmul_precision('high')


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args: OmegaConf) -> None:

    set_manual_seed(random_seed=args.random_seed)

    dataset: str = args.dataset.lower()
    basename = get_basename(args=args)
    print(f"Basename: {basename}")
    saving_path = f"models/{args.dataset.lower()}/{basename}.pth"
    Path(saving_path).parent.mkdir(exist_ok=True, parents=True)


    """ LOAD DATA """
    dataset_name = args.dataset.lower()
    assert dataset_name in ['camelyon16', 'seedp-stomach', 'seedp-colon'], f"{dataset_name} is not a valid dataset"
    print(f"\nDataset: {dataset_name}")

    num_patches = args.num_preselected_patches
    preselected_csv_dir = f"./csv/preselected_dataset={dataset_name}_num-patches={num_patches}_seed={args.random_seed}.csv"
    print(f"Preselected samples from: {preselected_csv_dir} with Proximity level: {args.proximity_level}")
    train_df = load_preselected_samples(preselected_csv_dir)

    if ('lonely' in args.positive_pair_sampling.lower()):
        # first 4 columns are supposed to be "slide", "coords", "pivot_patch", and "neighbor_level_0" (pivot_patch)
        # train_df = train_df[train_df.columns[:-2]]
        proximity_levels = train_df.columns[4:].values   
        for level in proximity_levels:
            indeces = np.arange(train_df.shape[0])
            indeces = indeces[train_df['pivot_patch'] == train_df[level]]
            train_df.loc[indeces, level] = np.nan
        train_df = train_df.dropna().reset_index(drop=True)
        print(f"Removing duplicated positive pairs. Remaining number of pairs: {train_df.shape[0]}")

    data = train_df[['pivot_patch', f'neighbor_level_{args.proximity_level}']].values.tolist()
   

    """ POSITIVE PAIR SAMPLIN METHOD """     
    transforms_pivot = ContextSamplingTransforms(resize_int=args.patch_resize).train_transforms
    transforms_neighbor = ContextSamplingTransforms(resize_int=args.patch_resize).train_transforms
    dataset = PreselectedPairsDataset(data=data, transforms_pivot=transforms_pivot, transforms_neighbor=transforms_neighbor)
    positive_pairs_sampling_method = two_distortion_positive_pairs
   
     
    """ DATALOADER """
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True)
    _ = next(iter(train_loader))
    print("Data loaded")


    """ SSL MODEL """
    # BACKBONE
    print(f"\nSSL Method: {args.ssl_method} with {args.backbone_name} as backbone")     
    backbone, embedding_dim = get_backbone(
        backbone_name=args.backbone_name,
        img_size=args.patch_resize)
    
      
    # SSL METHOD
    if args.ssl_method.lower() == 'barlow_twins':
        from src.ssl_methods.barlow_twins import BarlowTwins
        ssl_model = BarlowTwins(
            backbone=backbone,
            embedding_dim=embedding_dim,
            projector=args.barlow_twins.projector,
            lambd=args.barlow_twins.lambd)
        
    elif args.ssl_method.lower() == 'byol':
        from src.ssl_methods.byol import BYOL
        ssl_model = BYOL(
            backbone=backbone,
            embedding_dim=embedding_dim,
            mlp_hidden_size=args.byol.mlp_hidden_size,
            projector_dim=args.byol.projector_dim,
            momentum=args.byol.momentum)
        ssl_model._initialize_target_network()

    elif args.ssl_method.lower() == 'simsiam':
        from src.ssl_methods.simsiam import SimSiam
        ssl_model = SimSiam(
            backbone=backbone,
            embedding_dim=embedding_dim,
            encoder_dim=args.simsiam.encoder_dim,
            pred_dim=args.simsiam.pred_dim)
        
    elif args.ssl_method.lower() == 'vicreg':
        from src.ssl_methods.vicreg import VICReg
        ssl_model = VICReg(
            backbone=backbone,
            embedding_dim=embedding_dim,
            projector_arch=args.vicreg.projector_arch,
            sim_coeff=args.vicreg.sim_coeff,
            std_coeff=args.vicreg.std_coeff,
            cov_coeff=args.vicreg.cov_coeff)
        
    elif args.ssl_method.lower() == 'dino':
        raise ValueError(f"{args.ssl_method.lower()} is not implemented yet")
    
    else:
        raise ValueError(f"{args.ssl_method} is not a valid ssl method") 
    

    """ SSL Module """
    callbacks = [RichProgressBar(leave=False)]   
    if args.use_lars_optimizer:
        print("Using LARS optimizer")
        from src.lars import LARS, AdjustLearningRate, exclude_bias_and_norm
        optimizer: torch.optim = LARS(
            params=ssl_model.parameters(),
            lr=args.lars.lr,
            weight_decay=args.lars.weight_decay,
            weight_decay_filter=exclude_bias_and_norm,
            lars_adaptation_filter=exclude_bias_and_norm)
        scheduler = None
        adjust_learning_rate = AdjustLearningRate(
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            base_lr=args.lars.base_lr,
            num_batches=len(train_loader)
        )
        callbacks.append(adjust_learning_rate)
    else:
        optimizer: torch.optim = hydra.utils.instantiate(args.optimizer, params=ssl_model.parameters())
        scheduler: torch.optim.lr_scheduler = hydra.utils.instantiate(args.scheduler, optimizer=optimizer, T_max=args.max_epochs)
    
    use_moving_average: bool = args.ssl_method.lower() in ['byol']  
    ssl_module = SSLModule(
        ssl_model=ssl_model,
        positive_pairs_sampling_method=positive_pairs_sampling_method,
        optimizer=optimizer,
        scheduler=scheduler,
        saving_path=saving_path,
        use_moving_average=use_moving_average)


    """ TRAINING """
    start_time = time.perf_counter()

    print("\nTraining...")
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        precision="bf16-mixed",
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=callbacks,
        limit_val_batches=0)
    trainer.fit(model=ssl_module, train_dataloaders=train_loader)

    all_losses = {
        'step_losses': ssl_module.step_losses,
        'epoch_losses': ssl_module.epoch_losses
    }
    losses_path = f"local/ssl_losses/{dataset_name}/{basename}"
    Path(losses_path).parents[0].mkdir(exist_ok=True, parents=True)
    with open(f"{losses_path}.pkl", 'wb') as f:
        pickle.dump(all_losses, f, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed_time = time.perf_counter() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"SSL training took {int(hours)}h:{int(minutes)}m:{seconds:.2f}s.\n")

    return


if __name__=='__main__':
    main()
