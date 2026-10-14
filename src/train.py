"""Model training module for the Ames Housing regression pipeline.

Trains and tunes:
- Baseline Ordinary Least Squares Linear Regression.
- L2-Regularized Ridge Regression with Cross-Validated Alpha Tuning (RidgeCV).

Encapsulates complete preprocessing + estimator pipelines and handles model persistence.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_pipeline(preprocessor: ColumnTransformer, regressor: Any) -> Pipeline:
    """Combine ColumnTransformer preprocessing and an estimator into an end-to-end Pipeline.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Unfitted ColumnTransformer instance.
    regressor : Any
        Scikit-learn regressor (e.g., LinearRegression or Ridge).

    Returns
    -------
    Pipeline
        Unfitted full pipeline.
    """
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )


def train_baseline_linear_regression(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Train the baseline Ordinary Least Squares (OLS) Linear Regression model.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        ColumnTransformer for feature transformations.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target (SalePrice).

    Returns
    -------
    Pipeline
        Fitted baseline pipeline.
    """
    logger.info("Training Baseline LinearRegression model on %d training samples...", len(y_train))
    pipeline = build_pipeline(preprocessor, LinearRegression())
    pipeline.fit(X_train, y_train)
    logger.info("Baseline LinearRegression training complete.")
    return pipeline


def train_ridge_regression(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alphas: list[float] | np.ndarray | None = None,
    cv: int = 5,
) -> Tuple[Pipeline, float]:
    """Train Ridge regression with Cross-Validation tuning for the alpha hyperparameter.

    The Ridge Alpha Parameter:
    --------------------------
    Ridge regression minimizes the objective:
        ||y - Xw||^2_2 + alpha * ||w||^2_2

    The hyperparameter `alpha` controls the strength of the L2 penalty:
    - alpha = 0: Equivalent to Ordinary Least Squares (OLS), susceptible to overfitting
      and unstable weights in high-dimensional or collinear settings.
    - alpha > 0: Penalizes large coefficient magnitudes, shrinking them towards zero.
      This introduces a small bias while significantly reducing variance, stabilizing
      predictions against correlated housing features (e.g., GrLivArea, TotalBsmtSF, OverallQual).
    - Very large alpha: Over-penalizes parameters, causing underfitting.

    By using K-fold Cross-Validation on the training set exclusively, we select the
    optimal alpha that maximizes out-of-fold generalization without test set leakage.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        ColumnTransformer for feature transformations.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    alphas : list[float] | np.ndarray | None
        Grid of candidate alpha values to evaluate.
    cv : int
        Number of cross-validation folds. Defaults to 5.

    Returns
    -------
    Tuple[Pipeline, float]
        (fitted_pipeline, best_alpha)
    """
    if alphas is None:
        alphas = np.logspace(-2, 4, 30)

    logger.info(
        "Tuning Ridge regression using %d-fold CV over %d candidate alphas: [%.2e ... %.2e]",
        cv,
        len(alphas),
        float(alphas[0]),
        float(alphas[-1]),
    )

    ridge_cv = RidgeCV(alphas=alphas, cv=cv, scoring="neg_root_mean_squared_error")
    pipeline = build_pipeline(preprocessor, ridge_cv)
    pipeline.fit(X_train, y_train)

    best_alpha = float(pipeline.named_steps["regressor"].alpha_)
    logger.info("Optimal Ridge alpha selected via %d-fold CV: %.4f", cv, best_alpha)

    # Wrap in final standard Ridge model with the tuned alpha for clean inference
    final_ridge = Ridge(alpha=best_alpha, random_state=42)
    final_pipeline = build_pipeline(preprocessor, final_ridge)
    final_pipeline.fit(X_train, y_train)

    return final_pipeline, best_alpha


def save_pipeline(pipeline: Pipeline, filepath: str = "models/ridge_pipeline.joblib") -> str:
    """Serialize and save the complete fitted pipeline to disk using Joblib.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted Scikit-learn Pipeline (encapsulating preprocessor + regressor).
    filepath : str
        Target destination path for .joblib file.

    Returns
    -------
    str
        Absolute path to saved model file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    joblib.dump(pipeline, filepath)
    abs_path = os.path.abspath(filepath)
    logger.info("Successfully persisted complete model pipeline to: %s", abs_path)
    return abs_path
