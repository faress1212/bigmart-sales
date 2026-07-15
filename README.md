# Big Mart Sales — Visualization & Prediction 🛒

An end-to-end project analyzing and predicting Big Mart item-outlet sales: data cleaning, exploratory visualization, feature engineering, and a comparison of 4 regression models.

## Project Overview
This project builds a complete pipeline around the Big Mart Sales dataset. It covers:

- Data cleaning (missing values, inconsistent category labels)
- Exploratory Data Analysis: distributions, boxplots, bar charts, scatter plots, correlation heatmap
- Feature engineering (`outlet_age`, `item_visibility_scaled`, categorical encoding)
- Training and comparing 4 regression models to predict `item_outlet_sales`
- Saving trained models and generated figures to disk

## Project Structure
```
bigmart-sales-visualization/
├── data/
│   ├── raw/                        # Place Train.csv here
│   └── README.md
├── src/
│   ├── data/
│   │   └── load_data.py            # Loading + data quality checks + cleaning
│   ├── features/
│   │   └── preprocessing.py        # Feature engineering + train/test split
│   ├── models/
│   │   └── train_models.py         # Model training, evaluation, saving
│   ├── visualization/
│   │   └── eda_plots.py            # EDA plots (line, bar, histogram, box, violin, scatter, bubble, heatmap)
│   └── utils/
│       └── config.py               # Paths and shared settings (plain Python, no YAML)
├── models/                          # Saved trained models (.pkl)
├── reports/figures/                 # Saved EDA plots
├── main.py                          # Runs the full pipeline end-to-end
├── requirements.txt
└── README.md
```

## Dataset
The Big Mart Sales dataset (Analytics Vidhya / Kaggle) contains 8,523 rows covering 1,559 products across 10 outlets, collected in 2013. See `data/README.md` for the download link and column descriptions.

## Prediction Target
`item_outlet_sales` — the sales figure for a given item at a given outlet. This is a **regression** problem (predicting a continuous value), not classification.

## Models Compared
| Model               | Description                          |
|---------------------|---------------------------------------|
| Linear Regression   | Simple linear baseline                |
| Ridge Regression    | Linear model with L2 regularization   |
| Decision Tree       | Single tree-based regressor           |
| Random Forest       | Ensemble of decision trees            |

Models are evaluated using:
- **RMSE** (Root Mean Squared Error) — average error, same units as sales
- **MAE** (Mean Absolute Error) — average absolute error, less sensitive to outliers
- **R²** (coefficient of determination) — proportion of variance explained

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/bigmart-sales-visualization.git
cd bigmart-sales-visualization
```

### 2. Download the dataset
Download `Train.csv` from [Kaggle](https://www.kaggle.com/datasets/shivan118/big-mart-sales-prediction-datasets) and place it at `data/raw/Train.csv`.

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the pipeline
```bash
python main.py
```

This will:
- Load and clean the data
- Generate EDA plots into `reports/figures/`
- Train and evaluate 4 regression models
- Save each trained model into `models/`

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

## License

This project is open source and available under the [MIT License](LICENSE).
