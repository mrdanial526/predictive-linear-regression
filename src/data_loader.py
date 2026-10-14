"""Data ingestion and validation module for the Ames Housing regression pipeline.

This module provides robust loading from OpenML with automated local caching,
dataset structural inspection, and clean separation of features from target.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_data(
    data_path: str = "data/ames_housing.csv",
    cache: bool = True,
    openml_name: str = "house_prices",
    openml_version: int | str = "active",
) -> pd.DataFrame:
    """Load the Ames Housing dataset from local cache or OpenML.

    Parameters
    ----------
    data_path : str
        Path where local cached CSV file resides or will be saved.
    cache : bool
        Whether to cache newly fetched OpenML data locally.
    openml_name : str
        OpenML dataset name or identifier.
    openml_version : int | str
        Dataset version on OpenML.

    Returns
    -------
    pd.DataFrame
        Complete DataFrame containing raw Ames Housing features and target.
    """
    if os.path.exists(data_path):
        logger.info("Loading Ames Housing dataset from local cache: %s", data_path)
        df = pd.read_csv(data_path)
        return df

    logger.info("Local cache not found. Fetching '%s' from OpenML...", openml_name)
    try:
        dataset = fetch_openml(name=openml_name, version=openml_version, as_frame=True, parser="auto")
        df = dataset.frame.copy()
    except Exception as exc:
        logger.warning("Primary fetch failed (%s). Attempting OpenML dataset id 42165...", exc)
        dataset = fetch_openml(data_id=42165, as_frame=True, parser="auto")
        df = dataset.frame.copy()

    logger.info("Successfully fetched %d rows and %d columns from OpenML.", df.shape[0], df.shape[1])

    if cache:
        os.makedirs(os.path.dirname(os.path.abspath(data_path)), exist_ok=True)
        df.to_csv(data_path, index=False)
        logger.info("Dataset cached locally to: %s", data_path)

    return df


def inspect_data(df: pd.DataFrame) -> dict[str, Any]:
    """Inspect dataset dimensions, types, summary statistics, and missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to inspect.

    Returns
    -------
    dict[str, Any]
        Dictionary with inspection metrics.
    """
    n_rows, n_cols = df.shape
    dtypes_count = df.dtypes.value_counts().to_dict()
    missing_series = df.isnull().sum()
    missing_cols = missing_series[missing_series > 0].to_dict()
    total_missing = int(missing_series.sum())

    summary = {
        "num_rows": n_rows,
        "num_columns": n_cols,
        "dtype_counts": {str(k): v for k, v in dtypes_count.items()},
        "total_missing_values": total_missing,
        "columns_with_missing": len(missing_cols),
        "missing_breakdown": missing_cols,
    }

    logger.info("=== Dataset Inspection ===")
    logger.info("Dimensions: %d rows x %d columns", n_rows, n_cols)
    logger.info("Total Missing Values: %d across %d columns", total_missing, len(missing_cols))
    return summary


def separate_features_target(
    df: pd.DataFrame,
    target_col: str = "SalePrice",
    id_cols: list[str] | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Validate target variable, separate predictors from target, and remove IDs.

    Parameters
    ----------
    df : pd.DataFrame
        Complete dataset DataFrame.
    target_col : str
        Name of target column. Defaults to 'SalePrice'.
    id_cols : list[str] | None
        List of identifier columns to drop from feature set. Defaults to ['Id'].

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        (X, y) where X is the predictor feature DataFrame and y is the target Series.

    Raises
    ------
    KeyError
        If target_col is missing from DataFrame.
    ValueError
        If target_col has null values or DataFrame is empty.
    """
    if df.empty:
        raise ValueError("Cannot separate features from an empty DataFrame.")

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame columns: {list(df.columns)}")

    if df[target_col].isnull().any():
        null_count = int(df[target_col].isnull().sum())
        raise ValueError(f"Target column '{target_col}' contains {null_count} missing values.")

    if id_cols is None:
        id_cols = ["Id", "id"]

    cols_to_drop = [target_col] + [c for c in id_cols if c in df.columns]

    X = df.drop(columns=cols_to_drop).copy()
    y = df[target_col].astype(float).copy()

    logger.info(
        "Separated features and target: X shape=%s, y length=%d, dropped columns=%s",
        X.shape,
        len(y),
        [c for c in cols_to_drop if c != target_col],
    )
    return X, y
