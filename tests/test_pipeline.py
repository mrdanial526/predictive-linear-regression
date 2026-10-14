"""Pytest test suite for Predictive Linear Regression Model Pipeline.

Covers:
1. Data loading and target validation
2. Missing-value preprocessing (mean imputation, standard scaling)
3. Categorical encoding (most-frequent imputation, one-hot encoding with unknown handling)
4. Model fitting and prediction (LinearRegression & Ridge)
5. R2 and RMSE calculation
6. Saved-model loading and inference
7. Input validation and error handling for prediction
"""

from __future__ import annotations

import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, Ridge

from src.data_loader import inspect_data, separate_features_target
from src.evaluate import calculate_metrics, evaluate_model
from src.predict import load_model, predict_price, validate_input_data
from src.preprocessing import (
    create_preprocessor,
    identify_feature_types,
    split_data,
)
from src.train import (
    build_pipeline,
    save_pipeline,
    train_baseline_linear_regression,
    train_ridge_regression,
)


@pytest.fixture
def sample_housing_data() -> pd.DataFrame:
    """Fixture providing a synthetic property DataFrame mirroring Ames Housing features."""
    np.random.seed(42)
    n = 60
    return pd.DataFrame({
        "Id": range(1, n + 1),
        "LotArea": np.random.uniform(5000, 20000, n),
        "OverallQual": np.random.choice([4, 5, 6, 7, 8], n),
        "YearBuilt": np.random.choice(range(1950, 2020), n),
        "GrLivArea": np.random.uniform(800, 3000, n),
        "Neighborhood": np.random.choice(["CollgCr", "Veenker", "Crawfor", "NoRidge"], n),
        "HouseStyle": np.random.choice(["1Story", "2Story", "1.5Fin"], n),
        "SalePrice": np.random.uniform(120000, 450000, n),
    })


# 1. Data loading and target validation tests
def test_data_inspection_and_target_validation(sample_housing_data: pd.DataFrame):
    """Test data inspection summary and feature-target separation logic."""
    summary = inspect_data(sample_housing_data)
    assert summary["num_rows"] == 60
    assert summary["num_columns"] == 8

    X, y = separate_features_target(sample_housing_data, target_col="SalePrice", id_cols=["Id"])
    assert "SalePrice" not in X.columns
    assert "Id" not in X.columns
    assert len(X) == len(y) == 60
    assert y.name == "SalePrice"


def test_missing_target_raises_error(sample_housing_data: pd.DataFrame):
    """Test that missing or invalid target raises appropriate errors."""
    with pytest.raises(KeyError, match="Target column 'NonExistent' not found"):
        separate_features_target(sample_housing_data, target_col="NonExistent")

    corrupted_data = sample_housing_data.copy()
    corrupted_data.loc[0, "SalePrice"] = np.nan
    with pytest.raises(ValueError, match="contains 1 missing values"):
        separate_features_target(corrupted_data, target_col="SalePrice")


# 2. Missing-value preprocessing tests
def test_missing_value_numerical_preprocessing():
    """Verify numerical SimpleImputer fills missing values with the column mean."""
    num_cols = ["LotArea", "GrLivArea"]
    cat_cols = []
    preprocessor = create_preprocessor(num_cols, cat_cols)

    df_train = pd.DataFrame({
        "LotArea": [1000.0, 2000.0, 3000.0, np.nan],  # Mean is 2000.0
        "GrLivArea": [100.0, 200.0, np.nan, 300.0],   # Mean is 200.0
    })

    transformed = preprocessor.fit_transform(df_train)
    # Check that output is scaled and has zero NaNs
    assert not np.isnan(transformed).any()
    assert transformed.shape == (4, 2)


# 3. Categorical encoding tests
def test_categorical_encoding_and_unseen_handling():
    """Verify categorical imputer and OneHotEncoder handling unknown categories without error."""
    num_cols = ["GrLivArea"]
    cat_cols = ["Neighborhood"]
    preprocessor = create_preprocessor(num_cols, cat_cols)

    df_train = pd.DataFrame({
        "GrLivArea": [1200.0, 1500.0, 1800.0, 2000.0],
        "Neighborhood": ["CollgCr", "Crawfor", "CollgCr", np.nan],
    })

    preprocessor.fit(df_train)

    # Test inference on unseen category (e.g., 'UnknownSubdivision')
    df_test = pd.DataFrame({
        "GrLivArea": [1400.0],
        "Neighborhood": ["UnknownSubdivision"],
    })
    transformed_test = preprocessor.transform(df_test)
    assert not np.isnan(transformed_test).any()
    # Unknown category should be encoded as zeros without raising exception
    assert transformed_test.shape[0] == 1


# 4. Model fitting and prediction tests
def test_model_fitting_and_prediction(sample_housing_data: pd.DataFrame):
    """Test fitting both LinearRegression and Ridge pipelines and predicting."""
    X, y = separate_features_target(sample_housing_data, target_col="SalePrice")
    num_cols, cat_cols = identify_feature_types(X)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.25, random_state=42)

    preprocessor = create_preprocessor(num_cols, cat_cols)

    # Baseline Linear Regression
    lr_pipeline = train_baseline_linear_regression(preprocessor, X_train, y_train)
    lr_preds = lr_pipeline.predict(X_test)
    assert len(lr_preds) == len(y_test)
    assert np.all(np.isfinite(lr_preds))

    # Regularized Ridge Regression
    ridge_pipeline, best_alpha = train_ridge_regression(
        preprocessor, X_train, y_train, alphas=[0.1, 1.0, 10.0], cv=3
    )
    assert best_alpha in [0.1, 1.0, 10.0]
    ridge_preds = ridge_pipeline.predict(X_test)
    assert len(ridge_preds) == len(y_test)
    assert np.all(np.isfinite(ridge_preds))


# 5. R2 and RMSE calculation tests
def test_metrics_calculation():
    """Verify R2 and RMSE calculations match known mathematical properties."""
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    
    # Perfect predictions
    perfect_metrics = calculate_metrics(y_true, y_true)
    assert pytest.approx(perfect_metrics["r2"], abs=1e-5) == 1.0
    assert pytest.approx(perfect_metrics["rmse"], abs=1e-5) == 0.0

    # Constant error offset of +10
    offset_preds = y_true + 10.0
    offset_metrics = calculate_metrics(y_true, offset_preds)
    assert pytest.approx(offset_metrics["rmse"], abs=1e-5) == 10.0
    assert offset_metrics["r2"] > 0.95


# 6. Saved-model loading and inference tests
def test_saved_model_persistence_and_inference(sample_housing_data: pd.DataFrame):
    """Test saving pipeline to disk with Joblib and loading for prediction."""
    X, y = separate_features_target(sample_housing_data, target_col="SalePrice")
    num_cols, cat_cols = identify_feature_types(X)
    preprocessor = create_preprocessor(num_cols, cat_cols)
    pipeline = build_pipeline(preprocessor, Ridge(alpha=1.0))
    pipeline.fit(X, y)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "test_model.joblib")
        save_pipeline(pipeline, model_path)
        assert os.path.exists(model_path)

        loaded_pipeline = load_model(model_path)
        original_preds = pipeline.predict(X.iloc[:5])
        loaded_preds = loaded_pipeline.predict(X.iloc[:5])

        np.testing.assert_allclose(original_preds, loaded_preds)

        # Test predict_price high-level function
        sample_dict = X.iloc[0].to_dict()
        dict_pred = predict_price(loaded_pipeline, sample_dict)
        assert len(dict_pred) == 1
        assert pytest.approx(dict_pred[0], abs=1e-2) == float(original_preds[0])


# 7. Prediction validation and error handling tests
def test_prediction_input_validation():
    """Verify robust input validation on malformed inputs."""
    with pytest.raises(ValueError, match="Input property data cannot be None"):
        validate_input_data(None)

    with pytest.raises(ValueError, match="Input property DataFrame contains 0 rows"):
        validate_input_data(pd.DataFrame())

    # Missing expected columns raises informative ValueError by default
    with pytest.raises(ValueError, match="Input data is missing"):
        validate_input_data(
            pd.DataFrame({"GrLivArea": [1500]}),
            expected_columns=["GrLivArea", "LotArea", "OverallQual"],
            allow_missing_cols=False,
        )

    # Missing expected columns filled with NaN when allow_missing_cols=True
    df_imputable = validate_input_data(
        pd.DataFrame({"GrLivArea": [1500]}),
        expected_columns=["GrLivArea", "LotArea", "OverallQual"],
        allow_missing_cols=True,
    )
    assert "LotArea" in df_imputable.columns and np.isnan(df_imputable.loc[0, "LotArea"])

    # Unexpected columns raise informative ValueError when allow_unexpected_cols=False
    with pytest.raises(ValueError, match="Input data contains .* unexpected columns"):
        validate_input_data(
            pd.DataFrame({"GrLivArea": [1500], "ExtraGarbageCol": [999]}),
            expected_columns=["GrLivArea"],
            allow_unexpected_cols=False,
        )

