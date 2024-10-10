from src.utils_common import set_manual_seed
from src.utils_data import preselect_positive_pairs
from omegaconf import OmegaConf
import hydra
import pandas as pd

pd.options.mode.chained_assignment = None


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args: OmegaConf):

    set_manual_seed(random_seed=args.random_seed)


    """ LOAD DATASET """
    print(f"\nDataset: {args.dataset}")
    if args.dataset.lower() == 'camelyon16':
        from src.utils_data import get_camelyon16
        df = get_camelyon16()
    else:
        raise ValueError(f"{args.dataset} is not a valid dataset")
    
      
    """ SELECT POSITIVE PAIRS BASED ON PROXIMITY """
    for proximity_level in [0, 1, 2, 3, 4, 5, 10, 20]:
        print(f"Proximity level: {proximity_level}")
        train_df = df.query("subset=='train'")
        data = preselect_positive_pairs(
            train_df=train_df,
            proximity_level=proximity_level,
            batch_size=args.batch_size,
            num_batches=args.num_batches,
            random_seed=args.random_seed,
            same_patch_ratio=0.1)
        new_df = pd.DataFrame(data=data, columns=['positive_pairs', 'slide', 'coords'])
        new_df.to_csv(
            f"./csv/preselected_dataset={args.dataset.lower()}_level={proximity_level}_seed={args.random_seed}.csv",
            index=False)
        print()


    return


if __name__=='__main__':
    main()
