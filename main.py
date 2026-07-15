"""
End-to-end pipeline for the Big Mart Sales Visualization + Prediction project.

Requires data/raw/Train.csv (see data/README.md for download instructions).

Run with:
    python main.py
"""

from src.data.load_data import add_scaled_visibility, check_data_quality, clean_data, load_data
from src.features.preprocessing import add_outlet_age, encode_categorical_columns, split_and_scale
from src.models.train_models import get_models, save_model, train_and_evaluate
from src.utils.config import TARGET_COLUMN, ensure_dirs
from src.visualization.eda_plots import generate_all_plots


def main():
    ensure_dirs()

    # 1. Load & inspect data
    df = load_data()
    check_data_quality(df)

    # 2. Clean data
    df = clean_data(df)
    df = add_scaled_visibility(df)

    print("\nCleaned shape:", df.shape)
    print("Columns:", list(df.columns))

    # 3. EDA plots (uses the cleaned, not-yet-encoded dataframe so category
    #    labels read naturally)
    generate_all_plots(df)

    # 4. Feature engineering
    df_model = add_outlet_age(df)
    df_model = encode_categorical_columns(df_model)

    # 5. Train / test split + scaling
    X = df_model.drop(TARGET_COLUMN, axis=1)
    y = df_model[TARGET_COLUMN]
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

    # 6. Train & evaluate regression models
    models = get_models()
    results = train_and_evaluate(models, X_train, X_test, y_train, y_test)

    print("\nModel evaluation (lower RMSE/MAE is better, higher R^2 is better):")
    for name, metrics in results.items():
        print(
            f"{name}: RMSE={metrics['rmse']:.2f}  "
            f"MAE={metrics['mae']:.2f}  R^2={metrics['r2']:.4f}"
        )

    best_model_name = max(results, key=lambda name: results[name]["r2"])
    print(f"\nBest model (by R^2): {best_model_name} ({results[best_model_name]['r2']:.4f})")

    # 7. Save all trained models to disk
    print("\nSaving models:")
    for name, model in models.items():
        path = save_model(model, name)
        print(f"  {name} -> {path}")


if __name__ == "__main__":
    main()
