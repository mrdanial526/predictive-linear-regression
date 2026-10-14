"""Model evaluation and Exploratory Data Analysis (EDA) module.

Computes regression metrics (R2 and RMSE), generates structured comparison tables,
produces actual-vs-predicted and residual plots, and generates EDA visualizations.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Configure visualization styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.titlesize": 14,
    "figure.dpi": 150,
})


def calculate_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    """Calculate R2 score and Root Mean Squared Error (RMSE).

    Interpretation:
    - R2 Score: Proportion of variance in the dependent variable explained by the model.
      Values range from (-inf, 1.0]. A higher R2 score is preferable, with 1.0 indicating
      perfect explanation of variance.
    - RMSE: Square root of the average squared errors. Measures standard deviation of
      residuals in the same units as the target ($). A lower RMSE is preferable as it
      represents smaller prediction discrepancies.

    Parameters
    ----------
    y_true : Sequence[float]
        Ground truth continuous values.
    y_pred : Sequence[float]
        Model predicted values.

    Returns
    -------
    dict[str, float]
        {"r2": float, "rmse": float}
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    r2 = float(r2_score(y_true_arr, y_pred_arr))
    mse = float(mean_squared_error(y_true_arr, y_pred_arr))
    rmse = float(np.sqrt(mse))

    return {"r2": r2, "rmse": rmse}


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
) -> dict[str, Any]:
    """Evaluate a trained pipeline on held-out test data.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted Scikit-learn Pipeline.
    X_test : pd.DataFrame
        Held-out test features.
    y_test : pd.Series
        Held-out test target values.
    model_name : str
        Human-readable name of model.

    Returns
    -------
    dict[str, Any]
        Dictionary containing model_name, r2, rmse, y_pred, residuals.
    """
    logger.info("Evaluating %s on %d held-out test samples...", model_name, len(y_test))
    y_pred = pipeline.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)
    residuals = np.asarray(y_test, dtype=float) - np.asarray(y_pred, dtype=float)

    result = {
        "model_name": model_name,
        "r2": metrics["r2"],
        "rmse": metrics["rmse"],
        "y_pred": y_pred,
        "residuals": residuals,
    }

    logger.info(
        "[%s Evaluation] R2: %.4f | RMSE: $%.2f",
        model_name,
        metrics["r2"],
        metrics["rmse"],
    )
    return result


def compare_models(
    evaluations: list[dict[str, Any]],
    output_csv: str = "reports/metrics.csv",
) -> pd.DataFrame:
    """Compare multiple model evaluation results in a structured table and save to CSV.

    Parameters
    ----------
    evaluations : list[dict[str, Any]]
        List of evaluation dictionaries from evaluate_model().
    output_csv : str
        Filepath for CSV export.

    Returns
    -------
    pd.DataFrame
        Formatted comparison DataFrame.
    """
    records = []
    for ev in evaluations:
        records.append({
            "Model": ev["model_name"],
            "R2_Score": round(ev["r2"], 4),
            "RMSE": round(ev["rmse"], 2),
        })

    df_metrics = pd.DataFrame(records)

    # Save to reports directory
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df_metrics.to_csv(output_csv, index=False)
    logger.info("Saved structured metrics comparison table to: %s", os.path.abspath(output_csv))
    return df_metrics


def plot_actual_vs_predicted(
    evaluations: list[dict[str, Any]],
    y_test: pd.Series,
    output_path: str = "reports/actual_vs_predicted.png",
) -> None:
    """Generate and save Actual vs. Predicted scatter plot for evaluated models."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    n_models = len(evaluations)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5.5), sharey=True)
    if n_models == 1:
        axes = [axes]

    y_test_arr = np.asarray(y_test, dtype=float)
    min_val = min(y_test_arr.min(), min(ev["y_pred"].min() for ev in evaluations))
    max_val = max(y_test_arr.max(), max(ev["y_pred"].max() for ev in evaluations))

    for ax, ev in zip(axes, evaluations):
        ax.scatter(
            y_test_arr,
            ev["y_pred"],
            alpha=0.6,
            color="#2563EB",
            edgecolors="none",
            s=40,
            label="Predictions",
        )
        # Identity line y = x (perfect prediction)
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.8, label="Ideal (y = x)")
        ax.set_title(f"{ev['model_name']}\n$R^2 = {ev['r2']:.4f}$ | RMSE = ${ev['rmse']:,.0f}")
        ax.set_xlabel("Actual SalePrice ($)")
        ax.set_ylabel("Predicted SalePrice ($)")
        ax.legend(loc="upper left")
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved actual vs. predicted plot to: %s", os.path.abspath(output_path))


def plot_residuals(
    evaluations: list[dict[str, Any]],
    output_path: str = "reports/residual_plot.png",
) -> None:
    """Generate and save Residuals vs. Predicted scatter plot for error diagnostics."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    n_models = len(evaluations)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5.5), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, ev in zip(axes, evaluations):
        ax.scatter(
            ev["y_pred"],
            ev["residuals"],
            alpha=0.6,
            color="#059669",
            edgecolors="none",
            s=40,
        )
        ax.axhline(0, color="crimson", linestyle="--", linewidth=1.8)
        ax.set_title(f"{ev['model_name']} Residuals\nMean Res: ${np.mean(ev['residuals']):.1f}")
        ax.set_xlabel("Predicted SalePrice ($)")
        ax.set_ylabel("Residual (Actual - Predicted) ($)")
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved residual plot to: %s", os.path.abspath(output_path))


def run_eda(
    df: pd.DataFrame,
    target_col: str = "SalePrice",
    reports_dir: str = "reports",
) -> None:
    """Perform Exploratory Data Analysis and save diagnostic visualizations.

    Generates:
    - Target variable distribution (histogram + KDE).
    - Top numerical features correlation heatmap.
    - Missing value breakdown.

    Parameters
    ----------
    df : pd.DataFrame
        Complete dataset DataFrame.
    target_col : str
        Name of target column. Defaults to 'SalePrice'.
    reports_dir : str
        Directory to store visual artifacts.
    """
    os.makedirs(reports_dir, exist_ok=True)
    logger.info("Running Exploratory Data Analysis (EDA)...")

    # 1. Target Distribution Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[target_col], kde=True, color="#3B82F6", ax=ax, bins=40)
    ax.set_title(f"Target Distribution: {target_col}")
    ax.set_xlabel(f"{target_col} ($)")
    ax.set_ylabel("Frequency")
    median_val = df[target_col].median()
    mean_val = df[target_col].mean()
    ax.axvline(median_val, color="red", linestyle="--", label=f"Median: ${median_val:,.0f}")
    ax.axvline(mean_val, color="green", linestyle=":", label=f"Mean: ${mean_val:,.0f}")
    ax.legend()
    dist_path = os.path.join(reports_dir, "eda_target_distribution.png")
    plt.tight_layout()
    plt.savefig(dist_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved target distribution plot to %s", dist_path)

    # 2. Correlation with Target Heatmap (Top 12 numerical features)
    numeric_df = df.select_dtypes(include=[np.number])
    if target_col in numeric_df.columns:
        corr_series = numeric_df.corr()[target_col].abs().sort_values(ascending=False)
        top_corr_features = corr_series.head(12).index
        corr_matrix = numeric_df[top_corr_features].corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            square=True,
            ax=ax,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(f"Top Features Correlated with {target_col}")
        corr_path = os.path.join(reports_dir, "eda_correlation_matrix.png")
        plt.tight_layout()
        plt.savefig(corr_path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved correlation matrix to %s", corr_path)

    # 3. Missing Values Breakdown Plot
    missing = df.isnull().sum()
    missing_nonzero = missing[missing > 0].sort_values(ascending=False)
    if not missing_nonzero.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        missing_df = pd.DataFrame({
            "Feature": missing_nonzero.index,
            "MissingCount": missing_nonzero.values,
            "Percentage": (missing_nonzero.values / len(df)) * 100,
        })
        sns.barplot(
            data=missing_df.head(15),
            x="Percentage",
            y="Feature",
            hue="Feature",
            palette="Blues_r",
            legend=False,
            ax=ax,
        )
        ax.set_title("Top Features with Missing Values (%)")
        ax.set_xlabel("Missing Percentage (%)")
        missing_path = os.path.join(reports_dir, "eda_missing_values.png")
        plt.tight_layout()
        plt.savefig(missing_path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved missing values plot to %s", missing_path)
