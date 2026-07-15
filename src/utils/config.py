"""
Central configuration for the Big Mart Sales Visualization project.

Simple constants module (no YAML) holding file paths and shared settings
used across the pipeline.
"""

import os

# --- Project paths -----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
TRAIN_CSV_PATH = os.path.join(RAW_DATA_DIR, "Train.csv")

MODELS_DIR = os.path.join(BASE_DIR, "models")

REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

# --- Data settings -------------------------------------------------------
TARGET_COLUMN = "item_outlet_sales"
CATEGORICAL_COLUMNS = [
    "item_identifier",
    "item_fat_content",
    "item_type",
    "outlet_identifier",
    "outlet_size",
    "outlet_location_type",
    "outlet_type",
]

# --- Model settings ------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2


def ensure_dirs() -> None:
    """Create all project output directories if they don't already exist."""
    for directory in (RAW_DATA_DIR, MODELS_DIR, FIGURES_DIR):
        os.makedirs(directory, exist_ok=True)
