"""End-to-end orchestration runner for Ames Housing Predictive Linear Regression.

Executes:
1. Ingestion of Ames Housing dataset from OpenML / cache
2. Dataset structure inspection and target separation
3. Exploratory Data Analysis (EDA) visualizations
4. Leakage-free train/test split
5. Preprocessing pipeline construction (ColumnTransformer)
6. Model training: Baseline Linear Regression vs. Cross-Validated Ridge Regression
7. Held-out test set evaluation, structured metrics table, and diagnostic plots
8. Pipeline persistence via Joblib
9. Sample inference validation
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

from src.data_loader import inspect_data, load_data, separate_features_target
from src.evaluate import (
    compare_models,
    evaluate_model,
    plot_actual_vs_predicted,
    plot_residuals,
    run_eda,
)
from src.predict import load_model, predict_price
from src.preprocessing import (
    create_preprocessor,
    identify_feature_types,
    split_data,
)
from src.train import (
    save_pipeline,
    train_baseline_linear_regression,
    train_ridge_regression,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("AmesHousingPipeline")


def run_pipeline(
    data_path: str = "data/ames_housing.csv",
    models_dir: str = "models",
    reports_dir: str = "reports",
    random_state: int = 42,
    skip_eda: bool = False,
) -> dict:
    """Run the complete supervised regression pipeline.

    Parameters
    ----------
    data_path : str
        Path to cached or newly fetched CSV dataset.
    models_dir : str
        Directory to store serialized model artifacts.
    reports_dir : str
        Directory to store CSV metrics and PNG plots.
    random_state : int
        Seed for reproducibility across splits and modeling.
    skip_eda : bool
        Whether to skip generating EDA plots.

    Returns
    -------
    dict
        Summary dictionary of pipeline execution artifacts and metrics.
    """
    logger.info("==================================================================")
    logger.info("   STARTING AMES HOUSING PREDICTIVE LINEAR REGRESSION PIPELINE    ")
    logger.info("==================================================================")

    # 1. Ingestion
    logger.info("\n--- STEP 1: DATA INGESTION ---")
    df = load_data(data_path=data_path, cache=True)
    inspection = inspect_data(df)

    # 2. Exploratory Data Analysis (EDA)
    if not skip_eda:
        logger.info("\n--- STEP 2: EXPLORATORY DATA ANALYSIS (EDA) ---")
        run_eda(df, target_col="SalePrice", reports_dir=reports_dir)

    # 3. Separation of Features and Target
    logger.info("\n--- STEP 3: FEATURE & TARGET SEPARATION ---")
    X, y = separate_features_target(df, target_col="SalePrice", id_cols=["Id", "id"])
    numerical_cols, categorical_cols = identify_feature_types(X)

    # 4. Leakage-Free Train/Test Split
    logger.info("\n--- STEP 4: DATA SPLITTING ---")
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=0.2, random_state=random_state
    )

    # 5. Preprocessing Setup
    logger.info("\n--- STEP 5: PREPROCESSING PIPELINE SPECIFICATION ---")
    preprocessor = create_preprocessor(numerical_cols, categorical_cols)
    logger.info(
        "Configured ColumnTransformer with %d numerical (mean imputer + standard scaler) "
        "and %d categorical (most_frequent imputer + one-hot encoder) features.",
        len(numerical_cols),
        len(categorical_cols),
    )

    # 6. Model Training & Tuning
    logger.info("\n--- STEP 6: MODEL TRAINING & TUNING ---")
    # Baseline: Ordinary Least Squares Linear Regression
    lr_pipeline = train_baseline_linear_regression(preprocessor, X_train, y_train)

    # Regularized: Ridge Regression with Cross-Validated Alpha Tuning
    candidate_alphas = np.logspace(-2, 4, 30)
    ridge_pipeline, best_alpha = train_ridge_regression(
        preprocessor, X_train, y_train, alphas=candidate_alphas, cv=5
    )

    # 7. Model Evaluation on Held-Out Test Set
    logger.info("\n--- STEP 7: EVALUATION ON HELD-OUT TEST SET ---")
    eval_lr = evaluate_model(lr_pipeline, X_test, y_test, model_name="Baseline Linear Regression")
    eval_ridge = evaluate_model(
        ridge_pipeline, X_test, y_test, model_name=f"Ridge Regression (alpha={best_alpha:.2f})"
    )

    evaluations = [eval_lr, eval_ridge]

    # Save metrics table
    metrics_csv = os.path.join(reports_dir, "metrics.csv")
    metrics_df = compare_models(evaluations, output_csv=metrics_csv)
    print("\n" + "=" * 65)
    print("                 MODEL EVALUATION COMPARISON")
    print("=" * 65)
    print(metrics_df.to_string(index=False))
    print("=" * 65)
    print("Interpretation:")
    print(" - R2 Score: Higher is better (1.0 indicates perfect variance explanation).")
    print(" - RMSE: Lower is better (measures prediction error standard deviation in $).")
    print("=" * 65 + "\n")

    # Generate Evaluation Diagnostic Visualizations
    actual_vs_pred_png = os.path.join(reports_dir, "actual_vs_predicted.png")
    residuals_png = os.path.join(reports_dir, "residual_plot.png")
    plot_actual_vs_predicted(evaluations, y_test, output_path=actual_vs_pred_png)
    plot_residuals(evaluations, output_path=residuals_png)

    # 8. Model Persistence
    logger.info("\n--- STEP 8: MODEL SERIALIZATION ---")
    model_artifact_path = os.path.join(models_dir, "ridge_pipeline.joblib")
    baseline_artifact_path = os.path.join(models_dir, "baseline_pipeline.joblib")
    save_pipeline(ridge_pipeline, model_artifact_path)
    save_pipeline(lr_pipeline, baseline_artifact_path)


    # 9. Verification via Prediction Function
    logger.info("\n--- STEP 9: SAMPLE INFERENCE VERIFICATION ---")
    persisted_model = load_model(model_artifact_path)
    sample_records = X_test.iloc[:3].to_dict(orient="records")
    sample_preds = predict_price(persisted_model, sample_records)
    actual_sample_prices = y_test.iloc[:3].values

    for idx, (pred, actual) in enumerate(zip(sample_preds, actual_sample_prices), 1):
        diff = pred - actual
        logger.info(
            "Sample %d -> Predicted: $%-10.2f | Actual: $%-10.2f | Residual: $%+.2f",
            idx,
            pred,
            actual,
            diff,
        )

    logger.info("==================================================================")
    logger.info("         PIPELINE EXECUTION SUCCESSFULLY COMPLETED!              ")
    logger.info("==================================================================")

    return {
        "metrics": metrics_df,
        "best_alpha": best_alpha,
        "model_artifact": model_artifact_path,
        "metrics_csv": metrics_csv,
    }


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Predictive Linear Regression Pipeline on Ames Housing")
    parser.add_argument("--data-path", default="data/ames_housing.csv", help="Path to raw/cached CSV")
    parser.add_argument("--models-dir", default="models", help="Directory for model binaries")
    parser.add_argument("--reports-dir", default="reports", help="Directory for metric reports & plots")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-eda", action="store_true", help="Skip EDA plots generation")
    args = parser.parse_args()

    run_pipeline(
        data_path=args.data_path,
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
        random_state=args.seed,
        skip_eda=args.skip_eda,
    )


if __name__ == "__main__":
    main()
