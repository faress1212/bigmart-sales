"""
Model training and evaluation utilities for the Big Mart Sales
Visualization project.

Predicting 'item_outlet_sales' is a regression problem, so models and
metrics here are regression-based (no class balancing / SMOTE applies).
"""

import os
import pickle

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

from src.utils.config import MODELS_DIR, RANDOM_STATE


def get_models(random_state: int = RANDOM_STATE) -> dict:
    """
    Build the dictionary of candidate regression models.

    Args:
        random_state (int): Random seed for reproducibility.

    Returns:
        dict: Mapping of model name -> unfitted scikit-learn estimator.
    """
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(random_state=random_state),
        "Decision Tree": DecisionTreeRegressor(random_state=random_state),
        "Random Forest": RandomForestRegressor(random_state=random_state),
    }


def train_and_evaluate(models: dict, X_train, X_test, y_train, y_test) -> dict:
    """
    Fit each model and compute regression metrics on the test set.

    Metrics reported per model:
    - RMSE (Root Mean Squared Error): average error in the same units as sales
    - MAE (Mean Absolute Error): average absolute error, less sensitive to outliers
    - R^2 (coefficient of determination): proportion of variance explained

    Args:
        models (dict): Mapping of model name -> unfitted estimator.
        X_train, X_test, y_train, y_test: Train/test split data.

    Returns:
        dict: Mapping of model name -> dict of {"rmse", "mae", "r2"}.
    """
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[name] = {"rmse": rmse, "mae": mae, "r2": r2}

    return results


def save_model(model, name: str, models_dir: str = MODELS_DIR) -> str:
    """
    Save a fitted model to disk as a .pkl file.

    Args:
        model: Fitted scikit-learn estimator.
        name (str): Model name, used to build the output filename.
        models_dir (str): Directory to save the model into.

    Returns:
        str: Path to the saved model file.
    """
    os.makedirs(models_dir, exist_ok=True)
    safe_name = name.lower().replace(" ", "_")
    path = os.path.join(models_dir, f"{safe_name}.pkl")

    with open(path, "wb") as f:
        pickle.dump(model, f)

    return path
