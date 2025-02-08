from src.utils_common import set_manual_seed, get_basename
from src.utils_data import load_pivot_dataframe
from src.utils_transforms import StandardTransforms 
from src.utils_models import get_backbone
from lightning.pytorch.callbacks import RichProgressBar
from omegaconf import OmegaConf
from pathlib import Path
import time
import torch
import hydra
import pickle
import lightning as L


torch.set_float32_matmul_precision('high')


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args: OmegaConf) -> None:

    pps = args.positive_pair_sampling
    print(f"Positive pair sampling: {pps}")

    assert args.positive_pair_sampling in [
        'baseline', 
        'closest_patch', 
        'random_neighbor', 
        'different_slide_same_label',
        'different_normal_slide',
        'preselected_positive_pairs'], f"{pps} is not implemented"   

    set_manual_seed(random_seed=args.random_seed)

    basename = get_basename(args=args)
    print(f"Basename: {basename}")
    saving_path = f"models/{args.dataset.lower()}/{basename}.pth"
    Path(saving_path).parent.mkdir(exist_ok=True, parents=True)


    """ LOAD DATA """
    dataset_name = args.dataset.lower()
    # assert dataset_name in ['camelyon16', 'seedp-stomach', 'seedp-colon', 'tcga'], f"{dataset_name} is not a valid dataset"
    print(f"\nDataset: {dataset_name}")    
    # df = load_pivot_dataframe(dataset_name=dataset_name)
    # print("DataFrame is loaded")
    # # data = process_neighbor_data(df)  
    # data = df[['pivot_patch', 'slide', 'coords', 'slide_label', 'neighbors']].values
    # del df


    
    """ POSITIVE PAIR SAMPLIN METHOD """
    img_size = args.patch_resize if args.ssl_method.lower() != 'dino' else args.dino.img_size
    transforms = StandardTransforms(resize_int=img_size)

    if pps == 'baseline':
        from src.utils_self_supervised import BasicSSLDataset       
        from src.utils_positive_pairs import two_distortion_positive_pairs

        columns = ['pivot_patch']
        df = load_pivot_dataframe(dataset_name=dataset_name, columns=columns)
        print("DataFrame is loaded")
        # data = process_neighbor_data(df)  
        data = df[columns].values
        del df

        dataset = BasicSSLDataset(data=data, transforms=transforms)
        positive_pairs_sampling_method = two_distortion_positive_pairs

    elif pps == 'closest_patch':
        from src.utils_self_supervised import NeighborDataset
        from src.utils_positive_pairs import three_views_positive_pairs

        columns = ['pivot_patch', 'neighbors']
        df = load_pivot_dataframe(dataset_name=dataset_name, columns=columns)
        print("DataFrame is loaded")
        # data = process_neighbor_data(df)  
        data = df[columns].values
        del df

        assert args.maximum_distance is not None
        print(f"Selecting closest patch as Context Pair. Minimum distance: {args.maximum_distance}")
        dataset = NeighborDataset(data=data, minimum_distance=args.minimum_distance, maximum_distance=args.maximum_distance, transforms=transforms)        
        positive_pairs_sampling_method = three_views_positive_pairs

    elif pps == 'random_neighbor':
        from src.utils_self_supervised import NeighborDataset
        from src.utils_positive_pairs import three_views_positive_pairs
        dataset = NeighborDataset(data=data, minimum_distance=args.minimum_distance, maximum_distance=None, transforms=transforms)        
        positive_pairs_sampling_method = three_views_positive_pairs

    elif pps == 'different_normal_slide':
        from src.utils_data import get_normal_label
        from src.utils_self_supervised import TwoViewsDataset
        from src.utils_positive_pairs import ContextPositivePairs
        dataset = TwoViewsDataset(data=data, transforms=transforms)
        normal_label = get_normal_label(dataset=dataset_name)
        if normal_label is None:
            print("There is no Normal slide. Skipping...")
            return

        positive_pairs_sampling_method = ContextPositivePairs(normal_label=normal_label).diff_normal_slide_positive_pairs

    elif pps == 'different_slide_same_label':
        from src.utils_data import get_normal_label
        from src.utils_self_supervised import TwoViewsDataset
        from src.utils_positive_pairs import ContextPositivePairs
        dataset = TwoViewsDataset(data=data, transforms=transforms)
        positive_pairs_sampling_method = ContextPositivePairs().diff_slide_same_label_positive_pairs
    else:
        raise ValueError(f"{args.positive_pair_sampling} is not a valid positive pair sampling method")
    

    """ SSL MODEL """
    # BACKBONE
    print(f"\nSSL Method: {args.ssl_method} with {args.backbone_name} as backbone")     
    backbone, embedding_dim = get_backbone(
        backbone_name=args.backbone_name,
        img_size=img_size)
    
      
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
        use_distributed = len(args.devices) > 1
        ssl_model = VICReg(
            backbone=backbone,
            embedding_dim=embedding_dim,
            projector_arch=args.vicreg.projector_arch,
            sim_coeff=args.vicreg.sim_coeff,
            std_coeff=args.vicreg.std_coeff,
            cov_coeff=args.vicreg.cov_coeff,
            use_distributed=use_distributed)
        
    elif args.ssl_method.lower() == 'dino':
        print("Implementing DINO")
    
    else:
        raise ValueError(f"{args.ssl_method} is not a valid ssl method") 
    

    """ SSL Module """
    callbacks = [RichProgressBar(leave=False)]   

    if args.ssl_method.lower() != 'dino':
        from src.utils_self_supervised import SSLModule 

        train_loader = torch.utils.data.DataLoader(
            dataset=dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.num_workers,
            persistent_workers=True,
            pin_memory=True)
        one_batch = next(iter(train_loader))
        views = one_batch[0]
        print("Data loaded")
        print(f"Number of views: {len(views)}")
        del one_batch
        del views

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
            scheduler: torch.optim.lr_scheduler = hydra.utils.instantiate(
                args.scheduler, 
                optimizer=optimizer, 
                T_max=args.max_epochs)
        
        use_moving_average: bool = args.ssl_method.lower() in ['byol']  
        ssl_module = SSLModule(
            ssl_model=ssl_model,
            positive_pairs_sampling_method=positive_pairs_sampling_method,
            alpha=args.alpha,
            optimizer=optimizer,
            scheduler=scheduler,
            saving_path=saving_path,
            use_moving_average=use_moving_average)
        
    else:
        # DINO implementation
        print("Loading DINO Module")

        from src.utils_dino import DINOSSLModule
        from omegaconf import OmegaConf, open_dict

        hparams = args.dino
        OmegaConf.set_struct(hparams, True)
        with open_dict(hparams): 
            hparams.num_epochs = args.max_epochs
            hparams.num_workers = args.num_workers  
            hparams.num_gpus = len(args.devices)

        train_loader = torch.utils.data.DataLoader(
            dataset=dataset, 
            batch_size=hparams.batch_size, 
            shuffle=True, 
            num_workers=hparams.num_workers,
            persistent_workers=True,
            pin_memory=True)
        one_batch = next(iter(train_loader))
        views = one_batch[0]
        print("Data loaded")
        print(f"Number of views: {len(views)}")
        del one_batch
        del views

        with open_dict(hparams):
            hparams.niter_per_ep = len(train_loader)  

        ssl_module = DINOSSLModule(
            positive_pairs_sampling_method=positive_pairs_sampling_method,
            saving_path=saving_path,
            hparams=hparams)


    """ TRAINING """
    start_time = time.perf_counter()

    print("\nTraining...")
    num_gpus = len(args.devices)
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        precision="bf16-mixed",
        accelerator=args.accelerator,
        strategy='ddp' if num_gpus > 1 else 'auto',
        devices=num_gpus,
        callbacks=callbacks,
        limit_val_batches=0)
    trainer.fit(model=ssl_module, train_dataloaders=train_loader)

    all_losses = {
        'step_losses': ssl_module.step_losses,
        'epoch_losses': ssl_module.epoch_losses
    }
    losses_dir = f"local/ssl_losses/{dataset_name}"
    Path(losses_dir).mkdir(exist_ok=True, parents=True)
    with open(f"{losses_dir}/{basename}.pkl", 'wb') as f:
        pickle.dump(all_losses, f, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed_time = time.perf_counter() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"SSL training took {int(hours)}h:{int(minutes)}m:{seconds:.2f}s.\n")

    return


if __name__=='__main__':
    main()
