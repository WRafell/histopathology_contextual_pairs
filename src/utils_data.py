from pathlib import Path
from tqdm import tqdm
import random
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

    def extract_coords(img_path: list, patch_size: int):
        x, y = list(map(int, Path(img_path).stem.split('_')))
        return x//patch_size, y//patch_size

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Apply the coordinate extraction function to each image path and store in a new 'coords' column
    df['coords'] = df['img_path'].apply(lambda img_path: extract_coords(img_path, patch_size))

    return df

