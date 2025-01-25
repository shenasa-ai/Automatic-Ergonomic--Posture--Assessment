from sklearn.metrics import accuracy_score
import pandas as pd
from typing import Optional


def calculate_accuracy(
    pred_path: str,
    actual_path: str,
    pred_part: Optional[str] = None,
    actual_part: Optional[str] = None
) -> float:
    """
    Calculate the accuracy between predicted and actual labels.

    Parameters:
    - pred_path: Path to the CSV file containing predicted labels.
    - actual_path: Path to the CSV file containing actual labels.
    - pred_part: Column name in the predicted CSV to use for accuracy calculation. If None, uses all columns.
    - actual_part: Column name in the actual CSV to use for accuracy calculation. If None, uses all columns.

    Returns:
    - Accuracy score as a float.
    """
    # Load and sort data
    pred_df = pd.read_csv(pred_path).sort_values('image_number')
    actual_df = pd.read_csv(actual_path).sort_values('image_number')

    # Ensure the dataframes are aligned
    if not pred_df['image_number'].equals(actual_df['image_number']):
        raise ValueError("The 'image_number' columns in predicted and actual data do not match.")

    # Calculate accuracy
    if pred_part is None and actual_part is None:
        return accuracy_score(actual_df, pred_df)
    else:
        if pred_part is None or actual_part is None:
            raise ValueError("Both 'pred_part' and 'actual_part' must be provided if one is specified.")
        return accuracy_score(actual_df[actual_part], pred_df[pred_part])