import pandas as pd
import os
from typing import List


def clean_labels(
    labels_path: str,
    images_dir: str,
    output_path: str = 'final_labels_cleared.csv'
) -> None:
    """
    Cleans a labels CSV file by removing entries for images that no longer exist in the specified directory.

    Parameters:
    - labels_path: Path to the CSV file containing labeled image numbers.
    - images_dir: Path to the directory containing the image files.
    - output_path: Path to save the cleaned CSV file. Defaults to 'final_labels_cleared.csv'.
    """
    # Load the labels CSV file
    labels = pd.read_csv(labels_path, index_col=False)

    # Get the list of existing images
    exist_images = os.listdir(images_dir)

    # Extract image IDs from the filenames
    img_id = [int(i.split('.')[0].split('_')[-1]) for i in exist_images]

    # Get the list of labeled image numbers
    labeld_images = labels['image_number']

    # Identify images that are labeled but do not exist in the directory
    deleted_images = list(set(labeld_images) - set(img_id))

    # Drop rows corresponding to deleted images
    must_drop = labels[labels['image_number'].isin(deleted_images)].index
    labels.drop(index=must_drop, inplace=True)

    # Save the cleaned labels to a new CSV file
    labels.to_csv(output_path, index=False)
    print(f"Cleaned labels saved to {output_path}")
