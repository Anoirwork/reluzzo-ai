# utils.py
# Utility functions for the project

import pandas as pd

def load_data(file_path):
    """
    Load a CSV file into a pandas DataFrame.
    
    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    return pd.read_csv(file_path)