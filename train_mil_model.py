from src.utils_common import set_manual_seed
from omegaconf import OmegaConf
import torch
import hydra
import pandas as pd


torch.set_float32_matmul_precision('high')


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args: OmegaConf) -> None:

    set_manual_seed(random_seed=args.random_seed)
    basename = f"pps={args.positive_pair_sampling}_ssl={args.ssl_method}_encoder={args.backbone_name}"


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
    embedding_dim = 512 if args.backbone_name == 'resnet18' else 1024
    num_classes = 2 if args.dataset.lower() == 'camelyon16' else 3
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


    return


if __name__=='__main__':
    main()
