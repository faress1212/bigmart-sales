"""
Feature engineering and train/test split utilities for the Big Mart Sales
Visualization project.

No SMOTE / class-balancing here: this is a regression problem (predicting
a continuous sales figure), so class balancing doesn't apply. Kept simple
on purpose — label encoding + a plain train/test split.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils.config import CATEGORICAL_COLUMNS, RANDOM_STATE, TEST_SIZE


def add_outlet_age(df: pd.DataFrame, current_year: int = 2013) -> pd.DataFrame:
    """
    Add an 'outlet_age' feature: years since the outlet was established.

    The dataset was collected in 2013, so that's used as the reference year
    by default.

    Args:
        df (pd.DataFrame): Input dataframe with an 'outlet_establishment_year' column.
        current_year (int): Reference year to compute age against.

    Returns:
        pd.DataFrame: Dataframe with an added 'outlet_age' column.
    """
    df = df.copy()
    df["outlet_age"] = current_year - df["outlet_establishment_year"]
    return df


def encode_categorical_columns(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Label-encode categorical columns so they can be used by ML models.

    Note: a separate LabelEncoder is fit per column (rather than one shared
    encoder reused across columns), since each column has its own distinct
    set of categories.

    Args:
        df (pd.DataFrame): Input dataframe.
        columns (list, optional): Categorical column names to encode.
                                   Defaults to config.CATEGORICAL_COLUMNS.

    Returns:
        pd.DataFrame: Dataframe with the given columns label-encoded.
    """
    df = df.copy()
    columns = columns or CATEGORICAL_COLUMNS

    for col in columns:
        if col in df.columns:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))

    return df


def split_and_scale(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """
    Split features/target into train/test sets, then standardize features.

    The scaler is fit only on the training data and applied to both train
    and test sets, to avoid leaking test-set statistics into training.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: X_train_scaled, X_test_scaled, y_train, y_test, fitted_scaler
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
