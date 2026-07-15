"""
Data loading and cleaning utilities for the Big Mart Sales
Visualization project.

Dataset: Big Mart Sales dataset (Analytics Vidhya / Kaggle)
https://www.kaggle.com/datasets/shivan118/big-mart-sales-prediction-datasets
"""

import pandas as pd

from src.utils.config import TRAIN_CSV_PATH


def load_data(path: str = TRAIN_CSV_PATH) -> pd.DataFrame:
    """
    Load the Big Mart Sales training dataset from a CSV file.

    Args:
        path (str): Path to Train.csv. Defaults to data/raw/Train.csv
                     relative to the project root.

    Returns:
        pd.DataFrame: Raw sales dataframe.

    Raises:
        FileNotFoundError: If the dataset file doesn't exist at the given path,
                            with instructions on how to obtain it.
    """
    import os

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Download it from Kaggle: "
            "https://www.kaggle.com/datasets/shivan118/big-mart-sales-prediction-datasets "
            "and place 'Train.csv' inside the 'data/raw/' folder."
        )
    return pd.read_csv(path)


def check_data_quality(df: pd.DataFrame) -> None:
    """
    Print basic data quality checks: shape, info, missing values, summary stats.

    Args:
        df (pd.DataFrame): Input dataframe.
    """
    print("Shape:", df.shape)
    print("\nInfo:")
    df.info()
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nSummary statistics:")
    print(df.describe())


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataframe:
    - Standardize column names to lowercase, stripped of whitespace
    - Fill missing 'item_weight' with the mean
    - Fill missing 'outlet_size' with the mode
    - Standardize inconsistent 'item_fat_content' labels
      (e.g. 'LF', 'low fat' -> 'Low Fat'; 'reg' -> 'Regular')

    Args:
        df (pd.DataFrame): Raw sales dataframe.

    Returns:
        pd.DataFrame: Cleaned dataframe with lowercase column names.
    """
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()

    if "item_weight" in df.columns:
        df["item_weight"] = df["item_weight"].fillna(df["item_weight"].mean())

    if "outlet_size" in df.columns:
        df["outlet_size"] = df["outlet_size"].fillna(df["outlet_size"].mode()[0])

    if "item_fat_content" in df.columns:
        fat_content_map = {
            "LF": "Low Fat",
            "low fat": "Low Fat",
            "reg": "Regular",
        }
        df["item_fat_content"] = df["item_fat_content"].replace(fat_content_map)

    # Drop any remaining rows with missing values, just in case
    df = df.dropna(how="any")

    return df


def add_scaled_visibility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add an 'item_visibility_scaled' column (visibility * 100) for easier
    interpretation as a percentage.

    Args:
        df (pd.DataFrame): Cleaned dataframe with an 'item_visibility' column.

    Returns:
        pd.DataFrame: Dataframe with an added 'item_visibility_scaled' column.
    """
    df = df.copy()
    df["item_visibility_scaled"] = df["item_visibility"] * 100
    return df
