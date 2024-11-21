from src.utils_common import set_manual_seed, get_basename
from src.utils_self_supervised import SSLModule
from omegaconf import OmegaConf
from lightning.pytorch.callbacks import RichProgressBar
from pathlib import Path
import time
import torch
import hydra
import pickle
import torchvision
import lightning as L


torch.set_float32_matmul_precision('high')


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args: OmegaConf) -> None:

    assert args.positive_pair_sampling in [
        'baseline', 
        'closest_patch', 
        'random_neighbor', 
        'region',
        'different_slide_same_label',
        'different_normal_slide',
        'preselected_positive_pairs']
    
    set_manual_seed(random_seed=args.random_seed)

    basename = get_basename(args=args)
    print(f"Basename: {basename}")
    saving_path = f"models/{args.dataset.lower()}/{basename}.pth"
    Path(saving_path).parent.mkdir(exist_ok=True, parents=True)


    """ LOAD DATA """
    dataset_name = args.dataset.lower()
    assert dataset_name in ['camelyon16', 'seedp-stomach', 'seedp-colon'], f"{dataset_name} is not a valid dataset"
    print(f"\nDataset: {dataset_name}")

    preselected_samples = args.positive_pair_sampling == 'preselected_positive_pairs'
    if preselected_samples:
        from src.utils_data import load_preselected_samples
        from src.utils_self_supervised import PreselectedPairsDataset
        # num_patches = int(args.batch_size * args.num_batches)
        num_patches = args.num_preselected_patches
        preselected_csv_dir = f"./csv/preselected_dataset={dataset_name}_num-patches={num_patches}_seed={args.random_seed}.csv"
        print(f"Preselected samples from: {preselected_csv_dir} with Proximity level: {args.proximity_level}")
        train_df = load_preselected_samples(preselected_csv_dir)
        data = train_df[['pivot_patch', f'neighbor_level_{args.proximity_level}']].values.tolist()
    else:
        from src.utils_data import load_dataset, shuffle_train_data
        df = load_dataset(dataset_name)   
        train_df = df.query("subset=='train'")
        shuffled_train_df = shuffle_train_data(
            train_df=train_df,
            dataset_name=args.dataset.lower(),
            batch_size=args.batch_size,
            num_batches=args.num_batches,
            num_slides_per_batch=args.num_slides_per_batch)
        data = shuffled_train_df[['img_path', 'slide', 'coords', 'slide_label']].values.tolist()


    """ LOAD TRANSFORMS AND POSITIVE PAIR SAMPLIN METHOD """
    pps = args.positive_pair_sampling
    print(f"Positive pair sampling: {pps}")
    if pps == 'baseline':
        from src.utils_transforms import StandardTransforms 
        from src.utils_self_supervised import TwoViewsDataset       
        from src.utils_positive_pairs import two_distortion_positive_pairs
        transforms = StandardTransforms(resize_int=args.patch_resize).two_views_transform
        dataset = TwoViewsDataset(data=data, transforms=transforms)
        positive_pairs_sampling_method = two_distortion_positive_pairs

    elif pps == 'closest_patch':
        from src.utils_transforms import ContextSamplingTransforms
        from src.utils_self_supervised import TwoViewsDataset
        from src.utils_positive_pairs import ContextPositivePairs
        transforms = ContextSamplingTransforms(resize_int=args.patch_resize).two_views_transform
        dataset = TwoViewsDataset(data=data, transforms=transforms)
        print(f"Selecting closest patch as Context Pair. Minimum distance: {args.minimum_distance}")
        positive_pairs_sampling_method = ContextPositivePairs(minimum_distance=args.minimum_distance).closest_patch_positive_pairs

    elif pps == 'random_neighbor':
        from src.utils_transforms import ContextSamplingTransforms
        from src.utils_self_supervised import TwoViewsDataset
        from src.utils_positive_pairs import ContextPositivePairs
        transforms = ContextSamplingTransforms(resize_int=args.patch_resize).two_views_transform
        dataset = TwoViewsDataset(data=data, transforms=transforms)
        positive_pairs_sampling_method = ContextPositivePairs().random_patch_positive_pairs

    elif pps == 'different_normal_slide':
        from src.utils_data import get_normal_label
        from src.utils_transforms import ContextSamplingTransforms
        from src.utils_self_supervised import TwoViewsDataset
        from src.utils_positive_pairs import ContextPositivePairs
        transforms = ContextSamplingTransforms(resize_int=args.patch_resize).two_views_transform
        dataset = TwoViewsDataset(data=data, transforms=transforms)
        normal_label = get_normal_label(dataset=dataset_name)
        positive_pairs_sampling_method = ContextPositivePairs(normal_label=normal_label).diff_normal_slide_positive_pairs

    elif pps == 'different_slide_same_label':
        from src.utils_data import get_normal_label
        from src.utils_transforms import ContextSamplingTransforms
        from src.utils_self_supervised import TwoViewsDataset
        from src.utils_positive_pairs import ContextPositivePairs
        transforms = ContextSamplingTransforms(resize_int=args.patch_resize).two_views_transform
        dataset = TwoViewsDataset(data=data, transforms=transforms)
        positive_pairs_sampling_method = ContextPositivePairs().diff_slide_same_label_positive_pairs

    elif pps == 'region':
        # from src.utils_transforms import RegionTransforms 
        # from src.utils_self_supervised import TwoViewsDataset       
        # from src.utils_positive_pairs import two_distortion_positive_pairs
        # transforms = RegionTransforms(resize_int=args.patch_resize).two_views_transform
        # dataset = TwoViewsDataset(data=data, transforms=transforms)
        # positive_pairs_sampling_method = two_distortion_positive_pairs

        # TO BE IMPLEMENTED PROPERLY

        raise ValueError(f"{args.positive_pair_sampling} is not implemented yet")

    elif pps == 'preselected_positive_pairs':
        from src.utils_self_supervised import PreselectedPairsDataset
        from src.utils_positive_pairs import two_distortion_positive_pairs        
        if args.proximity_level == 0:
            from src.utils_transforms import StandardTransforms 
            transforms_pivot = StandardTransforms(resize_int=args.patch_resize).train_transforms
            transforms_neighbor = StandardTransforms(resize_int=args.patch_resize).train_transforms_prime
        else:
            from src.utils_transforms import ContextSamplingTransforms
            transforms_pivot = ContextSamplingTransforms(resize_int=args.patch_resize).train_transforms
            transforms_neighbor = ContextSamplingTransforms(resize_int=args.patch_resize).train_transforms
        dataset = PreselectedPairsDataset(data=data, transforms_pivot=transforms_pivot, transforms_neighbor=transforms_neighbor)
        positive_pairs_sampling_method = two_distortion_positive_pairs

    else:
        raise ValueError(f"{args.positive_pair_sampling} is not a valid positive pair sampling method")
    
     
    """ DATALOADER """
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True)
    _ = next(iter(train_loader))
    print("Data loaded")


    """ SSL MODEL """
    # BACKBONE
    print(f"\nSSL Method: {args.ssl_method} with {args.backbone_name} as backbone")     
    if args.backbone_name.lower() == 'resnet18':
        backbone = torchvision.models.resnet18(weights='ResNet18_Weights.DEFAULT')
        embedding_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()
    else:
        raise ValueError(f"{args.backbone_name} is not a valid backbone model")
    
      
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
    else:
        raise ValueError(f"{args.ssl_method} is not a valid ssl method") 
    

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
        callbacks = [RichProgressBar(leave=False), adjust_learning_rate]
    else:
        optimizer: torch.optim = hydra.utils.instantiate(args.optimizer, params=ssl_model.parameters())
        scheduler: torch.optim.lr_scheduler = hydra.utils.instantiate(args.scheduler, optimizer=optimizer, T_max=args.max_epochs)
        callbacks = [RichProgressBar(leave=False)]

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
