from pathlib import Path
from tqdm import tqdm
import random
import numpy as np
import pandas as pd


def shuffle_train_data(train_df: pd.DataFrame, batch_size: int, num_batches: int, num_slides_per_batch: int) -> pd.DataFrame:
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
        train_df: pd.DataFrame, 
        proximity_level: int, 
        batch_size: int, 
        num_batches: int, 
        random_seed: int,
        same_patch_ratio: float) -> list:
    """
    Preselects positive pairs of patches based on proximity level and same_patch_ratio.
    
    Parameters:
    - train_df: DataFrame containing columns 'img_path', 'slide', and 'coords'.
    - proximity_level: Integer representing the maximum proximity level for selecting neighboring patches.
    - batch_size: Integer for the number of patches sampled per batch.
    - num_batches: Integer for the total number of batches to sample.
    - random_seed: Integer seed for reproducibility.
    - same_patch_ratio: Float between 0 and 1 that determines the probability of selecting the same patch.
    
    Returns:
    - A list of tuples, where each tuple contains pairs of patches, slide, and coordinates.
    """

    # Sampling random patches
    patch_list = train_df.img_path.tolist()  
    random.seed(random_seed)   
    patch_list = random.sample(patch_list, int(batch_size * num_batches))
    # Store the selected positive pairs
    data = []  
    for i in tqdm(range(num_batches)): 
        patches = patch_list[i * batch_size:(i+1) * batch_size]
        patch_df = train_df.query(f"img_path in {patches}")  
        
        for row in patch_df.itertuples():  
            # If proximity level is 0, select the patch itself
            if proximity_level == 0:
                selected_patch = row.img_path
            else:
                # Filter patches from the same slide
                subdf = train_df.query(f"slide=='{row.slide}'")
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

            # Add the pair and associated data to the output
            data.append([(row.img_path, selected_patch), row.slide, row.coords])

    return data


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

