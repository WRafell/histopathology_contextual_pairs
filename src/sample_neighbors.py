from src.utils_common import set_manual_seed
from src.utils_data import load_dataset
from omegaconf import DictConfig
from tqdm import tqdm
import hydra
import numpy as np
from pathlib import Path
import pandas as pd

pd.options.mode.chained_assignment = None


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(args: DictConfig):
    set_manual_seed(random_seed=args.random_seed)


    """ LOAD DATASET """
    print(f"\nDataset: {args.dataset}")
    df = load_dataset(args.dataset.lower(), args.data_root)


    """ SELECT PATCHES AND NEIGHBORS """
    train_df = df.query("subset=='train'").copy()

    # Handle sampling with safety checks
    # num_samples = min(int(args.num_samples), train_df.shape[0])
    num_samples = min(int(args.num_samples), len(train_df))
    sampled_df = train_df.iloc[np.random.choice(train_df.shape[0], num_samples, replace=False)]

    # Precompute slide data with patch limits
    slide_groups = train_df.groupby('slide')
    slide_data = {}
    for slide, group in slide_groups:
        slide_data[slide] = {
            'coords': np.stack(group.coords.values),
            'img_paths': group.img_path.values
        }

    # Preallocate result containers
    data = {
        'slide': [],
        'slide_label': [],
        'patch_label': [],
        'coords': [],
        'pivot_patch': [],
        'neighbors': []
    }
    max_proximity = args.max_proximity + 1

    for row in tqdm(sampled_df.itertuples(), total=sampled_df.shape[0]):
        slide_info = slide_data.get(row.slide)
        if not slide_info:
            continue

        # Vectorized proximity calculation
        current_coord = np.array(row.coords)
        diffs = np.abs(slide_info['coords'] - current_coord)
        proximities = diffs.max(axis=1)

        # Create proximity map with numpy
        proximity_map = {}
        for lvl in np.unique(proximities):
            proximity_map[lvl] = slide_info['img_paths'][proximities == lvl].tolist()

        # Build neighbors list with fallback to pivot
        neighbors = []
        for lvl in range(max_proximity):
            if lvl == 0:
                neighbors.append([row.img_path])
            else:
                neighbors.append(proximity_map.get(lvl, []))

        # Store results
        data['slide'].append(row.slide)
        data['slide_label'].append(row.slide_label)
        data['patch_label'].append(row.label)
        data['coords'].append(row.coords)
        data['pivot_patch'].append(row.img_path)
        data['neighbors'].append(neighbors)

    new_df = pd.DataFrame(data)
    csv_dir = Path(args.data_root) / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(csv_dir / f"sampled_{args.dataset.lower()}.csv", index=False)
    return


if __name__=='__main__':
    main()
