from src.utils_common import set_manual_seed, get_basename
from omegaconf import OmegaConf
from pathlib import Path
import torch
import hydra
import numpy as np
import pandas as pd


torch.set_float32_matmul_precision('high')

# To avoid NoneType error due to hydra
TEST_ACC, TEST_AUC = [], []
BASENAME = None
PROXIMITY_LEVEL = None
LOGGER_MODE = None
LOGGER_NAME = None
LOGGER_PATH = None


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args: OmegaConf):

    global TEST_ACC
    global TEST_AUC
    global BASENAME
    global LOGGER_MODE
    global LOGGER_NAME
    global LOGGER_PATH
    global PROXIMITY_LEVEL
    
    basename = get_basename(args=args)
    print(f"Basename: {basename}")
    BASENAME = basename
    PROXIMITY_LEVEL = args.proximity_level
    if (args.positive_pair_sampling.lower() == 'preselected_positive_pairs') and (args.proximity_level == 0):
        LOGGER_MODE = 'w'
    else:
        LOGGER_MODE = 'a'

    LOGGER_NAME = f"pps={args.positive_pair_sampling}_ssl={args.ssl_method}_encoder={args.backbone_name}"    

    if "level=" in basename:
        LOGGER_PATH = f"logs/slide_level/{args.dataset}/proximity_ablation"
    else:
        LOGGER_PATH = f"logs/slide_level/{args.dataset}/standard_exp"

    Path(LOGGER_PATH).mkdir(exist_ok=True, parents=True)

    print(f"Logger path: {LOGGER_PATH}. Logger mode: {LOGGER_MODE}")


    """ LOAD DATA """
    emb_file = f"csv/{args.dataset}_{basename}.csv"
    df = pd.read_csv(emb_file)
    train_df = df.query("subset=='train'")
    val_df = df.query("subset=='val'")
    test_df = df.query("subset=='test'")
    train_data = train_df[['emb_path', 'label']].values.tolist()
    val_data = val_df[['emb_path', 'label']].values.tolist()
    test_data = test_df[['emb_path', 'label']].values.tolist()


    """ SLIDE CLASSIFICATION """
    # Embedding dimension
    if args.backbone_name.lower() == 'resnet18':
        embedding_dim = 512
    else:
        raise ValueError(f"{args.backbone_name} or { args.ssl_method.lower()} are not implemented")
    
    # Number of classes    
    if args.dataset.lower() == 'camelyon16':
        num_classes = 2
    else:
        raise ValueError(f"{args.dataset.lower()} dataset is not implemented")
    
    # MIL method
    if args.mil.mil_method.lower() == 'abmil':
        from src.mil_methods.abmil import ABMIL
        trainer = ABMIL(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            learning_rate=args.mil.learning_rate,
            weight_decay=args.mil.weight_decay,
            max_epochs=args.mil.max_epochs,
            device=args.device)
    else:
        raise ValueError(f"{args.mil.mil_method.lower()} is not valid MIL method")
    
    best_model = trainer.train(train_data, val_data)
    test_loss, test_acc, test_auc = trainer.test(best_model, test_data)  
    print(f"Results from: {basename} bags. MIL model: {args.mil.mil_method.lower()}")  
    print(f"Test loss: {test_loss:.4f}. Test acc: {test_acc:.4f}. AUROC: {test_auc:.4f}")  

    TEST_ACC.append(test_acc)   
    TEST_AUC.append(test_auc) 

    return 


if __name__=='__main__':
    
    num_runs = 5
    for i in range(num_runs):
        print(f"Run: {i+1}/{num_runs}")
        set_manual_seed(random_seed=i)
        main()
        print()

    test_acc_mean = np.mean(TEST_ACC)
    test_acc_std = np.std(TEST_ACC)
    test_auc_mean = np.mean(TEST_AUC)
    test_auc_std = np.std(TEST_AUC)
    
    print(f"Test acc: {test_acc_mean:.4f} +/- {test_acc_std:.4f}. "
          f"AUROC: {test_auc_mean:.4f} +/- {test_auc_std:.4f}.")  

        
    with open(f"{LOGGER_PATH}/{LOGGER_NAME}.txt", LOGGER_MODE) as logger:
        print(f"{BASENAME}", file=logger)
        print(f"Level: {PROXIMITY_LEVEL}", file=logger)   
        print(f"All test acc: {TEST_ACC}", file=logger)
        print(f"All test auroc: {TEST_AUC}", file=logger)     
        print(f"Test acc: {test_acc_mean:.4f} +/- {test_acc_std:.4f}. "
          f"AUROC: {test_auc_mean:.4f} +/- {test_auc_std:.4f}.", 
          file=logger)   
        print("-" * 100, file=logger)
        print('\n', file=logger)


