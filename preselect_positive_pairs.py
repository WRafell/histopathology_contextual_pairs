from src.utils_common import set_manual_seed
from src.utils_data import preselect_positive_pairs
from omegaconf import OmegaConf
import hydra
import random
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
    # Selecting N patches, where N = batch_size * num_batches
    proximity_levels = [0, 1, 2, 3, 4, 5, 10, 20]
    print(f"Proximity levels: {proximity_levels}")

    train_df = df.query("subset=='train'")
    # num_patches = int(args.batch_size * args.num_batches)
    num_patches = args.num_preselected_patches
    new_train_df = preselect_positive_pairs(
        all_train_df=train_df,
        num_patches=num_patches,
        proximity_levels=proximity_levels,
        same_patch_ratio=0.1)
    new_train_df.to_csv(
        f"./csv/preselected_dataset={args.dataset.lower()}_num-patches={num_patches}_seed={args.random_seed}.csv",
        index=False)
    

    # """ SAVE CSV FILES """
    # for proximity_level in proximity_levels:
    #     subdf = new_train_df.copy()
    #     subdf['positive_pair'] = [(v1, v2) for (v1, v2) in subdf[['pivot_patch', f'neighbor_level_{proximity_level}']].values]
    #     to_save = subdf[['positive_pair', 'slide', 'coords']].reset_index(drop=True)
    #     to_save.to_csv(
    #         f"./csv/preselected_dataset={args.dataset.lower()}_level={proximity_level}_seed={args.random_seed}.csv",
    #         index=False)
    #     print()


    return


if __name__=='__main__':
    main()
