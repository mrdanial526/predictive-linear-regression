"""Data preprocessing module for Ames Housing regression pipeline.

Defines feature typing, numerical and categorical preprocessing pipelines,
ColumnTransformer composition, and leak-free train/test splitting.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def identify_feature_types(X: pd.DataFrame) -> Tuple[list[str], list[str]]:
    """Automatically identify numerical and categorical column names.

    Parameters
    ----------
    X : pd.DataFrame
        Predictor feature DataFrame.

    Returns
    -------
    Tuple[list[str], list[str]]
        (numerical_cols, categorical_cols)
    """
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    logger.info(
        "Identified %d numerical features and %d categorical features.",
        len(numerical_cols),
        len(categorical_cols),
    )
    return numerical_cols, categorical_cols


def create_preprocessor(
    numerical_cols: list[str],
    categorical_cols: list[str],
) -> ColumnTransformer:
    """Build a ColumnTransformer applying numerical and categorical pipelines.

    Functional Requirements:
    - Numerical features:
        1. SimpleImputer(strategy="mean")
        2. StandardScaler()
    - Categorical features:
        1. SimpleImputer(strategy="most_frequent")
        2. OneHotEncoder(handle_unknown="ignore")

    Parameters
    ----------
    numerical_cols : list[str]
        List of numerical feature names.
    categorical_cols : list[str]
        List of categorical feature names.

    Returns
    -------
    ColumnTransformer
        Configured, unfitted ColumnTransformer.
    """
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and target into train and test sets before preprocessing fitting.

    Fitting preprocessing exclusively on X_train prevents data leakage from the test set.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    test_size : float
        Proportion of data held out for testing. Defaults to 0.2.
    random_state : int
        Seed for reproducibility. Defaults to 42.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(
        "Split data: X_train=%s, X_test=%s, y_train=%d, y_test=%d (test_ratio=%.2f, seed=%d)",
        X_train.shape,
        X_test.shape,
        len(y_train),
        len(y_test),
        test_size,
        random_state,
    )
    return X_train, X_test, y_train, y_test
