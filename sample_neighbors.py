from src.utils_common import set_manual_seed
from src.utils_data import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm
import hydra
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args: OmegaConf):
    set_manual_seed(random_seed=args.random_seed)


    """ LOAD DATASET """
    print(f"\nDataset: {args.dataset}")
    df = load_dataset(args.dataset.lower())

    if isinstance(df.coords[0], str):
        df['coords'] = [eval(coord) for coord in df.coords]

    """ SELECT PATCHES AND NEIGHBORS """
    train_df = df.query("subset=='train'").copy()
    
    # Fix typo and use iloc for safer sampling
    num_samples = min(int(args.num_samples), train_df.shape[0])
    sampled_df = train_df.iloc[np.random.choice(train_df.shape[0], num_samples, replace=False)]

    # Precompute slide groups and coordinate arrays
    slide_groups = train_df.groupby('slide')
    slide_data = {
        slide: {
            'coords': np.stack(group['coords'].values),
            'img_paths': group['img_path'].values
        }
        for slide, group in slide_groups
    }

    # Preallocate result arrays
    proximity_levels = range(args.max_proximity + 1)
    data = {
        'slide': [],
        'coords': [],
        'pivot_patch': [],
        'neighbors': []
    }

    for row in tqdm(sampled_df.itertuples(), total=sampled_df.shape[0]):
        slide_info = slide_data.get(row.slide)
        if not slide_info:
            continue

        # Vectorized proximity calculation
        current_coord = np.array(row.coords)
        diffs = np.abs(slide_info['coords'] - current_coord)
        proximities = diffs.max(axis=1)
        
        # Create proximity map using numpy
        unique_proximities = np.unique(proximities)
        proximity_map = {
            lvl: slide_info['img_paths'][proximities == lvl].tolist()
            for lvl in unique_proximities
        }

        # Build neighbors list with fallback to pivot patch
        neighbors = []
        for lvl in proximity_levels:
            if lvl == 0:
                neighbors.append([row.img_path])
            else:
                neighbors.append(proximity_map.get(lvl, [row.img_path]))

        # Store results
        data['slide'].append(row.slide)
        data['coords'].append(row.coords)
        data['pivot_patch'].append(row.img_path)
        data['neighbors'].append(neighbors)

    new_df = pd.DataFrame(data)
    new_df.to_csv(f"csv/sampled_{args.dataset.lower()}.csv", index=False)
    return


if __name__=='__main__':
    main()
