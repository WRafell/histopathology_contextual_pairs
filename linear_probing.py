from src.utils_common import set_manual_seed, get_basename
from src.utils_data import load_dataset
from src.utils_supervised import Transforms, LabeledDataset, PatchLevelClassifier, PatchLevelTrainer
from src.utils_models import get_backbone
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from datetime import datetime
from pathlib import Path
import torch
import hydra
import numpy as np
import pandas as pd


torch.set_float32_matmul_precision('high')


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args: OmegaConf) -> None:

    PROXIMITY_LEVEL = None
    MINIMUM_DISTANCE = None
    LOGGER_MODE = 'a'
    LOGGER_NAME = None
    LOGGER_PATH = None
    
    basename = get_basename(args=args)
    print(f"Basename: {basename}")
    LOGGER_NAME = f"pps={args.positive_pair_sampling}_ssl={args.ssl_method}_encoder={args.backbone_name}"   
    if ('preselected_positive_pairs' in args.positive_pair_sampling.lower()):
        LOGGER_PATH = f"logs/patch_level/{args.dataset}/proximity_ablation"
        PROXIMITY_LEVEL = args.proximity_level
        if (args.proximity_level == 0):
            LOGGER_MODE = 'w'
    else:
        LOGGER_PATH = f"logs/patch_level/{args.dataset}/standard_exp"   

    if (args.positive_pair_sampling.lower() == 'closest_patch'):
        MINIMUM_DISTANCE = args.minimum_distance   

    if args.supervised.train_baseline:
        MINIMUM_DISTANCE = None
        PROXIMITY_LEVEL = None
        LOGGER_MODE = 'a'
        LOGGER_NAME = "supervised_baseline"
        LOGGER_PATH = f"logs/patch_level/{args.dataset}/standard_exp"   


    Path(LOGGER_PATH).mkdir(exist_ok=True, parents=True)
    print(f"Logger path: {LOGGER_PATH}. Logger mode: {LOGGER_MODE}")

    TEST_ACC = []
    TEST_AUC = []
    
    for run in range(args.supervised.num_runs):

        set_manual_seed(random_seed=run)

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
        train_set = LabeledDataset(dataset=dataset['train'], transforms=train_transforms)
        val_set = LabeledDataset(dataset=dataset['val'], transforms=test_transforms)
        test_set = LabeledDataset(dataset=dataset['test'], transforms=test_transforms)

        train_loader = DataLoader(
            dataset=train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True)

        val_loader = DataLoader(
            dataset=val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True)

        test_loader = DataLoader(
            dataset=test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True)

        if args.dataset.lower() in ['seedp-stomach', 'seedp-colon']:
            num_classes = 3
        elif args.dataset.lower() == 'camelyon16':
            num_classes = 2


        """ LOAD ENCODER """
        print(f"\nSSL Method: {args.ssl_method} with {args.backbone_name} as backbone")     
        encoder, embedding_dim = get_backbone(
            backbone_name=args.backbone_name,
            img_size=args.patch_resize)
        
        if not args.supervised.train_baseline:        
            encoder_path = f"models/{args.dataset.lower()}/{basename}.pth"
            print(f"Encoder path: {encoder_path}")
            assert Path(encoder_path).exists(), f"{encoder_path} does not exist"
            state_dict = torch.load(encoder_path, weights_only=True)
            encoder.load_state_dict(state_dict)
            for param in encoder.parameters():
                param.requires_grad = False

        model = PatchLevelClassifier(
            encoder=encoder, 
            embedding_dim=embedding_dim, 
            num_classes=num_classes)

        """ TRAINER """
        trainer = PatchLevelTrainer(
            model=model,
            learning_rate=args.supervised.learning_rate,
            max_epochs=args.supervised.max_epochs,
            device="cuda:0",
            patience=5)
        best_model = trainer.train(train_loader, val_loader)
        _, test_acc, test_auc = trainer.test(best_model, test_loader, num_classes)  
       
        TEST_ACC.append(test_acc)   
        TEST_AUC.append(test_auc) 

    test_acc_mean = np.mean(TEST_ACC)
    test_acc_std = np.std(TEST_ACC)
    test_auc_mean = np.mean(TEST_AUC)
    test_auc_std = np.std(TEST_AUC)

    print(f"Test acc: {test_acc_mean:.4f} +/- {test_acc_std:.4f}. "
          f"AUROC: {test_auc_mean:.4f} +/- {test_auc_std:.4f}.")  
    
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y/%m/%d %H:%M:%S") 
    with open(f"{LOGGER_PATH}/{LOGGER_NAME}.txt", LOGGER_MODE) as logger:
        print(f"{basename}", file=logger)
        print(formatted_time, file=logger)
        if PROXIMITY_LEVEL is not None:
            print(f"Level: {PROXIMITY_LEVEL}", file=logger)   
        if MINIMUM_DISTANCE is not None:
            print(f"Minimum Distance: {MINIMUM_DISTANCE}", file=logger)  
        print(f"All test acc: {TEST_ACC}", file=logger)
        print(f"All test auroc: {TEST_AUC}", file=logger)     
        print(f"Test acc: {test_acc_mean:.4f} +/- {test_acc_std:.4f}. "
          f"AUROC: {test_auc_mean:.4f} +/- {test_auc_std:.4f}.", 
          file=logger)   
        print("-" * 100, file=logger)
        print('\n', file=logger)


    return


if __name__=='__main__':
    main()
