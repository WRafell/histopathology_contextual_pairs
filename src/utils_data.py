from pathlib import Path
from tqdm import tqdm
import random
import numpy as np
import pandas as pd


def shuffle_train_data(
        train_df: pd.DataFrame, 
        batch_size: int, 
        num_batches: int, 
        num_slides_per_batch: int) -> pd.DataFrame:
    print("Manually shuffling training data")
    slides = train_df.slide.unique().tolist()
    new_train_df = None
    for i in tqdm(range(num_batches)):
        random_slides = random.sample(slides, num_slides_per_batch)
        subdf = train_df.query(f"slide in {random_slides}").sample(n=batch_size)
        subdf = subdf.sample(n=batch_size)
        new_train_df = pd.concat([new_train_df, subdf], axis=0) if new_train_df is not None else subdf
    return new_train_df


def preselect_positive_pairs(
        all_train_df: pd.DataFrame, 
        num_patches: int,
        proximity_levels: list, 
        same_patch_ratio: float) -> pd.DataFrame:
    
    """
    Preselect positive pairs of patches for a given set of proximity levels.

    Parameters:
    - all_train_df: DataFrame containing all available patches, which will be used for proximity selection.
    - num_patches: Number of patches to sample from all_train_df
    - proximity_levels: List of proximity levels to consider for selecting neighbors.
    - same_patch_ratio: Float between 0 and 1, determining the probability of selecting the same patch.

    Returns:
    - A DataFrame with columns for the pivot patch and its neighbors at each proximity level.
    """

    data = []  # List to store the selected positive pairs

    patch_list = all_train_df.img_path.tolist()  
    patch_list = random.sample(patch_list, num_patches)
    sampled_df = all_train_df.query(f"img_path in {patch_list}")  

    # Define new columns for the resulting DataFrame
    new_df_columns = ['slide', 'coords', 'pivot_patch']
    new_df_columns.extend([f"neighbor_level_{i}" for i in proximity_levels])

    for row in tqdm(sampled_df.itertuples(), total=sampled_df.shape[0], desc="Selecting neighbors"):
        new_row = [row.slide, row.coords, row.img_path]

        for proximity_level in proximity_levels:
            # If proximity level is 0, select the patch itself
            if proximity_level == 0:
                selected_patch = row.img_path
            else:
                # Filter patches from the same slide
                subdf = all_train_df.query(f"slide=='{row.slide}'")
                # Calculate proximity (relative distance) for each patch in the slide
                subdf['relative_coords'] = subdf['coords'].apply(
                    lambda coords: np.abs(np.array(coords) - np.array(row.coords)))
                subdf['proximity_level'] = np.stack(subdf.relative_coords.values).max(axis=1)

                # Select neighbors that match the desired proximity level
                neighbors = subdf.query(f"proximity_level == {proximity_level}")

                # If no neighbors are found, select the current patch; otherwise, randomly sample one neighbor
                if neighbors.empty:
                    selected_patch = row.img_path 
                else:
                    selected_patch = neighbors.sample(n=1).reset_index().img_path[0]

            # Apply same_patch_ratio: With some probability, use the same patch instead of a neighbor
            if np.random.rand() < same_patch_ratio:
                selected_patch = row.img_path

            new_row.append(selected_patch)

        data.append(new_row)

    return pd.DataFrame(data, columns=new_df_columns)


def get_camelyon16(csv_file: str = "csv/camelyon16_data.csv", patch_size: int = 256) -> list:
    """
    Load the Camelyon16 dataset from a CSV file, extracting coordinates from image file names and 
    returning the image paths, slide identifiers, and patch coordinates.

    Args:
        csv_file (str): Path to the CSV file containing the Camelyon16 dataset metadata. 
                        The CSV file should have at least 'img_path' and 'slide' columns.
        patch_size (int): Size of the patch used to compute coordinates. Defaults to 256.

    Returns:
        list: A list of tuples, each containing:
              - img_path (str): The path to the image.
              - slide (str): The slide identifier.
              - coords (tuple): The (x, y) coordinates of the patch in the slide, adjusted by `patch_size`.
    """

    def extract_coords(img_path: list):
        x, y = list(map(int, Path(img_path).stem.split('_')))
        return x//512, y//512

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Apply the coordinate extraction function to each image path and store in a new 'coords' column
    df['coords'] = df['img_path'].apply(lambda img_path: extract_coords(img_path))

    return df


def get_seedp_stomach():
    return pd.read_csv("csv/seedp-stomach_data.csv")


def get_seedp_colon():
    return pd.read_csv("csv/seedp-colon_data.csv")


def load_dataset(dataset: str) -> pd.DataFrame:
    if dataset == 'camelyon16':
        return get_camelyon16()
    elif dataset == 'seedp-stomach':
        return get_seedp_stomach()
    elif dataset == 'seedp-colon':
        return get_seedp_colon()
    else:
        raise ValueError(f"{dataset} is not a valid dataset")
    

def load_preselected_samples(csv_dir: str) -> pd.DataFrame:
    assert Path(csv_dir).exists(), f"{csv_dir} does not exist"
    return pd.read_csv(csv_dir)








