"""Inference and prediction module for Ames Housing property valuation.

Loads saved Joblib pipeline, validates incoming property records, and generates
continuous price predictions with robust error handling for missing/unexpected columns.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_model(filepath: str = "models/ridge_pipeline.joblib") -> Pipeline:
    """Load a serialized Scikit-learn Pipeline from disk.

    Parameters
    ----------
    filepath : str
        Path to .joblib model file.

    Returns
    -------
    Pipeline
        Fitted Scikit-learn Pipeline.

    Raises
    ------
    FileNotFoundError
        If model artifact does not exist.
    TypeError
        If loaded object is not a Pipeline.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at: {os.path.abspath(filepath)}")

    model = joblib.load(filepath)
    if not isinstance(model, Pipeline):
        raise TypeError(f"Loaded object must be a sklearn.pipeline.Pipeline, got {type(model)}")

    logger.info("Successfully loaded model pipeline from: %s", os.path.abspath(filepath))
    return model


def validate_input_data(
    input_data: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]],
    expected_columns: Sequence[str] | None = None,
    allow_missing_cols: bool = False,
    allow_unexpected_cols: bool = True,
) -> pd.DataFrame:
    """Validate incoming property data and convert to formatted DataFrame.

    Parameters
    ----------
    input_data : pd.DataFrame | Mapping | Sequence[Mapping]
        Input property features.
    expected_columns : Sequence[str] | None
        Expected feature names required by the pipeline.
    allow_missing_cols : bool
        If True, missing expected columns are filled with NaN to be imputed by the pipeline.
        If False (default), raises an informative ValueError.
    allow_unexpected_cols : bool
        If True (default), extraneous columns not recognized by the pipeline are dropped with a warning.
        If False, raises an informative ValueError.

    Returns
    -------
    pd.DataFrame
        Cleaned, validated input DataFrame ready for inference.

    Raises
    ------
    ValueError
        If input data is empty, structurally invalid, missing expected columns, or contains unexpected columns.
    TypeError
        If input data format is unsupported.
    """
    if input_data is None:
        raise ValueError("Input property data cannot be None.")

    if isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    elif isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, (list, tuple)):
        if not input_data:
            raise ValueError("Input property list cannot be empty.")
        df = pd.DataFrame(list(input_data))
    else:
        raise TypeError(
            f"Unsupported input data type: {type(input_data)}. Expected DataFrame, dict, or list of dicts."
        )

    if df.empty:
        raise ValueError("Input property DataFrame contains 0 rows.")

    if expected_columns is not None:
        missing_cols = [c for c in expected_columns if c not in df.columns]
        if missing_cols:
            if not allow_missing_cols:
                raise ValueError(
                    f"Input data is missing {len(missing_cols)} expected feature columns: {missing_cols[:10]}... "
                    "Ensure all expected property characteristics are supplied or set allow_missing_cols=True."
                )
            logger.warning(
                "Input data is missing %d columns; filling with NaN for pipeline imputation: %s",
                len(missing_cols),
                missing_cols[:10],
            )
            for c in missing_cols:
                df[c] = np.nan

        unexpected_cols = [c for c in df.columns if c not in expected_columns]
        if unexpected_cols:
            if not allow_unexpected_cols:
                raise ValueError(
                    f"Input data contains {len(unexpected_cols)} unexpected columns: {unexpected_cols[:10]}... "
                    "Remove unrecognized features or set allow_unexpected_cols=True."
                )
            logger.warning(
                "Input data contains %d unexpected columns that will be ignored: %s",
                len(unexpected_cols),
                unexpected_cols[:10],
            )
            # Retain only expected features
            df = df[[c for c in expected_columns if c in df.columns]]

    return df


def predict_price(
    pipeline: Pipeline,
    input_data: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]],
    allow_missing_cols: bool = False,
    allow_unexpected_cols: bool = True,
) -> np.ndarray:
    """Predict sale prices ($) for new property observations using a fitted pipeline.

    Applies identical preprocessing steps (imputation, scaling, one-hot encoding)
    encapsulated in the pipeline, preventing leakage or training-serving skew.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted Scikit-learn Pipeline.
    input_data : pd.DataFrame | Mapping | Sequence[Mapping]
        Input features for one or more properties.
    allow_missing_cols : bool
        If True, missing expected columns are filled with NaN to be imputed by the pipeline.
        If False (default), raises an informative ValueError.
    allow_unexpected_cols : bool
        If True (default), extraneous columns not recognized by the pipeline are dropped with a warning.
        If False, raises an informative ValueError.

    Returns
    -------
    np.ndarray
        Array of predicted house sale prices in US Dollars.
    """
    # Extract expected feature names from pipeline ColumnTransformer if available
    preprocessor = pipeline.named_steps.get("preprocessor")
    expected_cols = None
    if preprocessor is not None and hasattr(preprocessor, "feature_names_in_"):
        expected_cols = preprocessor.feature_names_in_

    df_validated = validate_input_data(
        input_data,
        expected_columns=expected_cols,
        allow_missing_cols=allow_missing_cols,
        allow_unexpected_cols=allow_unexpected_cols,
    )

    predictions = pipeline.predict(df_validated)
    predictions = np.asarray(predictions, dtype=float)

    logger.info("Generated %d price predictions. Range: [$%.2f, $%.2f]", len(predictions), predictions.min(), predictions.max())
    return predictions
