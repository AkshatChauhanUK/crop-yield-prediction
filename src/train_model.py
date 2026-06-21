"""
train_model.py
--------------
Trains and compares models to predict crop yield, using k-fold cross-validation
for more reliable metrics on this small dataset.

Models:
    1. Linear Regression (baseline)
    2. Random Forest Regressor
    3. XGBoost Regressor

Evaluation:
    5-fold cross-validated RMSE, MAE, R^2 (averaged across folds).

Run:
    python src/train_model.py --input data/cleaned_crop_data.csv --outdir outputs
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor


TARGET = "yield_quintal_hectare"
CATEGORICAL_FEATURES = ["crop", "state"]
NUMERIC_FEATURES = [
    "cost_of_cultivation_`_hectare_a2+fl",
    "cost_of_cultivation_`_hectare_c2",
    "cost_of_production_`_quintal_c2",
]


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def build_pipeline(model):
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="passthrough",
    )
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def cross_validate_model(model_name, pipeline, X, y, cv):
    neg_mse_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="neg_mean_squared_error")
    neg_mae_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="neg_mean_absolute_error")
    r2_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="r2")

    rmse_scores = np.sqrt(-neg_mse_scores)
    mae_scores = -neg_mae_scores

    print(f"\n{model_name} (5-fold Cross-Validation)")
    print(f"  RMSE: {rmse_scores.mean():.2f} (+/- {rmse_scores.std():.2f})")
    print(f"  MAE:  {mae_scores.mean():.2f} (+/- {mae_scores.std():.2f})")
    print(f"  R^2:  {r2_scores.mean():.3f} (+/- {r2_scores.std():.3f})")

    return {
        "model": model_name,
        "rmse_mean": rmse_scores.mean(),
        "rmse_std": rmse_scores.std(),
        "mae_mean": mae_scores.mean(),
        "mae_std": mae_scores.std(),
        "r2_mean": r2_scores.mean(),
        "r2_std": r2_scores.std(),
    }
def tune_random_forest(X, y, cv):
    """Finds the best Random Forest hyperparameters using grid search."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="passthrough",
    )
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(random_state=42)),
    ])

    param_grid = {
        "model__n_estimators": [50, 100, 200, 300],
        "model__max_depth": [3, 5, 10, None],
    }

    grid_search = GridSearchCV(
        pipeline, param_grid, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1
    )
    grid_search.fit(X, y)

    print("\n" + "=" * 60)
    print("RANDOM FOREST HYPERPARAMETER TUNING")
    print("=" * 60)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV MAE: {-grid_search.best_score_:.2f}")

    return grid_search.best_estimator_

def plot_predictions(results_dict, outdir):
    n = len(results_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, (y_true, y_pred)) in zip(axes, results_dict.items()):
        ax.scatter(y_true, y_pred, alpha=0.7)
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([0, max_val], [0, max_val], "r--", label="Perfect prediction")
        ax.set_xlabel("Actual Yield")
        ax.set_ylabel("Predicted Yield")
        ax.set_title(name)
        ax.legend()

    plt.tight_layout()
    path = os.path.join(outdir, "actual_vs_predicted.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nSaved: {path}")


def plot_feature_importance(rf_pipeline, outdir):
    preprocessor = rf_pipeline.named_steps["preprocessor"]
    model = rf_pipeline.named_steps["model"]

    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    all_feature_names = cat_feature_names + NUMERIC_FEATURES

    importances = model.feature_importances_

    importance_by_group = {"crop": 0.0, "state": 0.0}
    for name, imp in zip(all_feature_names, importances):
        if name.startswith("crop_"):
            importance_by_group["crop"] += imp
        elif name.startswith("state_"):
            importance_by_group["state"] += imp
        else:
            importance_by_group[name] = imp

    sorted_items = sorted(importance_by_group.items(), key=lambda x: x[1], reverse=True)
    labels = [x[0] for x in sorted_items]
    values = [x[1] for x in sorted_items]

    plt.figure(figsize=(8, 5))
    plt.barh(labels, values, color="seagreen")
    plt.xlabel("Importance (summed for categorical groups)")
    plt.title("Feature Importance (Random Forest)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    path = os.path.join(outdir, "feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_cv_comparison(cv_results_df, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(
        cv_results_df["model"], cv_results_df["rmse_mean"],
        yerr=cv_results_df["rmse_std"], capsize=5, color="steelblue"
    )
    axes[0].set_title("Cross-Validated RMSE (lower is better)")
    axes[0].set_ylabel("RMSE")
    axes[0].tick_params(axis="x", rotation=15)

    axes[1].bar(
        cv_results_df["model"], cv_results_df["mae_mean"],
        yerr=cv_results_df["mae_std"], capsize=5, color="darkorange"
    )
    axes[1].set_title("Cross-Validated MAE (lower is better)")
    axes[1].set_ylabel("MAE")
    axes[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    path = os.path.join(outdir, "cv_model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to cleaned_crop_data.csv")
    parser.add_argument("--outdir", required=True, help="Folder to save charts")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = load_data(args.input)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = df[TARGET]

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_results = []

    lr_pipeline = build_pipeline(LinearRegression())
    cv_results.append(cross_validate_model("Linear Regression", lr_pipeline, X, y, cv))

    rf_pipeline = build_pipeline(RandomForestRegressor(n_estimators=200, random_state=42))
    cv_results.append(cross_validate_model("Random Forest", rf_pipeline, X, y, cv))
    tuned_rf_pipeline = tune_random_forest(X, y, cv)
    cv_results.append(cross_validate_model("Random Forest (Tuned)", tuned_rf_pipeline, X, y, cv))

    xgb_pipeline = build_pipeline(
        XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
    )
    cv_results.append(cross_validate_model("XGBoost", xgb_pipeline, X, y, cv))

    cv_results_df = pd.DataFrame(cv_results)
    print("\n" + "=" * 60)
    print("CROSS-VALIDATED MODEL COMPARISON (5-fold)")
    print("=" * 60)
    print(cv_results_df.to_string(index=False))

    cv_results_path = os.path.join(args.outdir, "cv_model_comparison.csv")
    cv_results_df.to_csv(cv_results_path, index=False)
    print(f"\nSaved: {cv_results_path}")

    plot_cv_comparison(cv_results_df, args.outdir)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    predictions_for_plot = {}

    lr_pipeline.fit(X_train, y_train)
    predictions_for_plot["Linear Regression"] = (y_test, lr_pipeline.predict(X_test))

    rf_pipeline.fit(X_train, y_train)
    predictions_for_plot["Random Forest"] = (y_test, rf_pipeline.predict(X_test))

    xgb_pipeline.fit(X_train, y_train)
    predictions_for_plot["XGBoost"] = (y_test, xgb_pipeline.predict(X_test))

    plot_predictions(predictions_for_plot, args.outdir)
    plot_feature_importance(rf_pipeline, args.outdir)

    print("\nModel training complete.")


if __name__ == "__main__":
    main()