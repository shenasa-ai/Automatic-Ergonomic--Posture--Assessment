from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
import pandas as pd
from typing import List, Dict
import os

def FleissKappa(file_paths: List[str], columns: List[str] = ['chair 1-2', 'back support', 'monitor']) -> Dict[str, float]:
    """
    Calculate Fleiss' Kappa for inter-rater agreement across multiple CSV files.

    Parameters:
    - file_paths: List of file paths to CSV files containing rater data.
    - columns: List of column names to consider for Fleiss' Kappa calculation.

    Returns:
    - A dictionary with Fleiss' Kappa values for each specified column.
    """
    # Read all CSV files and extract the specified columns
    data_frames = [pd.read_csv(file, usecols=columns) for file in file_paths]

    # Find indices of rows to drop (where any column has a value of 0)
    rows_to_drop = data_frames[0][(data_frames[0] == 0).any(axis=1)].index

    # Drop the identified rows from all data frames
    data_frames = [df.drop(index=rows_to_drop) for df in data_frames]

    # Concatenate the specified columns from all data frames into numpy arrays
    kappa_results = {}
    for column in columns:
        combined_data = pd.concat([df[column] for df in data_frames], axis=1).to_numpy()
        aggregated_data, _ = aggregate_raters(combined_data)
        kappa_results[column] = fleiss_kappa(aggregated_data, method='fleiss')

    return kappa_results