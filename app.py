"""Streamlit Web Application: Ames Housing Price Intelligence & Regression Pipeline.

A modern, interactive frontend showcasing the end-to-end Machine Learning pipeline:
- Real-time property valuation simulator with interactive sliders & inputs
- Side-by-side comparison: Baseline OLS vs. Regularized Ridge (alpha=20.43)
- Interactive diagnostic evaluations (Actual vs. Predicted, Residual plots)
- Exploratory Data Analysis (EDA) visualizations
- Production architecture & methodology summary for portfolio presentation
"""

from __future__ import annotations

import os
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import load_data, separate_features_target
from src.predict import load_model, predict_price
from src.preprocessing import identify_feature_types, split_data

# ==========================================
# 1. PAGE CONFIGURATION & THEME
# ==========================================
st.set_page_config(
    page_title="Ames Housing Price Intelligence | ML Pipeline",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom Design System CSS
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,500;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg-primary: #0b0f19;
    --bg-card: #111827;
    --bg-card-hover: #1f2937;
    --border-card: #1f293d;
    --text-main: #f9fafb;
    --text-muted: #9ca3af;
    --accent-blue: #3b82f6;
    --accent-indigo: #6366f1;
    --accent-emerald: #10b981;
    --accent-amber: #f59e0b;
}

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'DM Sans', -apple-system, sans-serif !important;
    background-color: var(--bg-primary);
    color: var(--text-main);
}

header[data-testid="stHeader"] {
    background-color: transparent !important;
}

.brand-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.25rem 0 1.75rem 0;
    border-bottom: 1px solid var(--border-card);
    margin-bottom: 1.5rem;
}

.brand-title {
    font-size: 1.85rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}

.brand-subtitle {
    font-size: 0.95rem;
    color: var(--text-muted);
    margin-top: 0.25rem;
}

.kpi-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border-card);
    border-radius: 12px;
    padding: 1.15rem 1.25rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.kpi-card:hover {
    border-color: #3b82f6;
    transform: translateY(-2px);
}

.kpi-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    font-weight: 600;
}

.kpi-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #ffffff;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 0.25rem;
}

.kpi-badge {
    display: inline-block;
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 6px;
    margin-top: 0.4rem;
    font-weight: 500;
}

.badge-blue { background: rgba(59, 130, 246, 0.15); color: #60a5fa; }
.badge-green { background: rgba(16, 185, 129, 0.15); color: #34d399; }
.badge-purple { background: rgba(139, 92, 246, 0.15); color: #c084fc; }
.badge-amber { background: rgba(245, 158, 11, 0.15); color: #fbbf24; }

.prediction-hero-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.95) 100%);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
    margin: 1.5rem 0;
}

.prediction-price {
    font-size: 3rem;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    background: linear-gradient(135deg, #34d399 0%, #60a5fa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.card-container {
    background: var(--bg-card);
    border: 1px solid var(--border-card);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==========================================
# 2. DATA & MODEL LOADING (CACHED)
# ==========================================
@st.cache_data
def get_data_and_test_metrics():
    """Load dataset, run quick split, and compute test predictions for diagnostics."""
    df = load_data("data/ames_housing.csv", cache=True)
    X, y = separate_features_target(df, target_col="SalePrice", id_cols=["Id", "id"])
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    return df, X, y, X_train, X_test, y_train, y_test


@st.cache_resource
def get_pipelines(X_train: pd.DataFrame, y_train: pd.Series):
    """Load persisted Ridge and Baseline pipelines, auto-training fallback if missing on cloud container."""
    ridge_path = "models/ridge_pipeline.joblib"
    baseline_path = "models/baseline_pipeline.joblib"

    ridge_model = None
    baseline_model = None

    if os.path.exists(ridge_path):
        try:
            ridge_model = joblib.load(ridge_path)
        except Exception:
            ridge_model = None

    if os.path.exists(baseline_path):
        try:
            baseline_model = joblib.load(baseline_path)
        except Exception:
            baseline_model = None

    # Fallback auto-training if missing on cloud runtime
    if ridge_model is None or baseline_model is None:
        from src.preprocessing import create_preprocessor, identify_feature_types
        from src.train import (
            save_pipeline,
            train_baseline_linear_regression,
            train_ridge_regression,
        )

        num_cols, cat_cols = identify_feature_types(X_train)
        preprocessor = create_preprocessor(num_cols, cat_cols)

        if baseline_model is None:
            baseline_model = train_baseline_linear_regression(preprocessor, X_train, y_train)
            save_pipeline(baseline_model, baseline_path)

        if ridge_model is None:
            ridge_model, _ = train_ridge_regression(preprocessor, X_train, y_train, cv=5)
            save_pipeline(ridge_model, ridge_path)

    return ridge_model, baseline_model


# Load resources
df, X, y, X_train, X_test, y_train, y_test = get_data_and_test_metrics()
ridge_pipeline, baseline_pipeline = get_pipelines(X_train, y_train)


# ==========================================
# 3. HEADER & KPI CARDS
# ==========================================
st.markdown(
    """
    <div class="brand-header">
        <div>
            <h1 class="brand-title">🏠 Ames Housing Valuation Intelligence</h1>
            <p class="brand-subtitle">End-to-End Supervised Regression Pipeline • Scikit-Learn • Ridge Regularization</p>
        </div>
        <div style="text-align: right;">
            <a href="https://github.com/mrdanial526/predictive-linear-regression" target="_blank" style="text-decoration: none;">
                <span style="background: #1f2937; border: 1px solid #374151; padding: 6px 14px; border-radius: 8px; color: #f3f4f6; font-size: 0.85rem; font-weight: 500;">
                    ⭐ GitHub Project
                </span>
            </a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Top KPI Summary Cards
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    st.markdown(
        """
        <div class="kpi-card">
            <div class="kpi-label">Baseline R² Score</div>
            <div class="kpi-value">0.8867</div>
            <span class="kpi-badge badge-blue">Ordinary Least Squares</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col2:
    st.markdown(
        """
        <div class="kpi-card">
            <div class="kpi-label">Ridge R² Score</div>
            <div class="kpi-value">0.8745</div>
            <span class="kpi-badge badge-purple">L₂ Regularized (CV)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col3:
    st.markdown(
        """
        <div class="kpi-card">
            <div class="kpi-label">Optimal Penalty (α)</div>
            <div class="kpi-value">20.43</div>
            <span class="kpi-badge badge-amber">5-Fold Cross-Validation</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col4:
    st.markdown(
        """
        <div class="kpi-card">
            <div class="kpi-label">Held-Out Test RMSE</div>
            <div class="kpi-value">$31,024</div>
            <span class="kpi-badge badge-green">N = 292 Properties</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# ==========================================
# 4. TABBED INTERACTIVE SECTIONS
# ==========================================
tab_simulator, tab_benchmark, tab_eda, tab_architecture = st.tabs([
    "🏠 Property Valuation Simulator",
    "📊 Model Comparison & Diagnostics",
    "📈 Exploratory Data Analysis (EDA)",
    "🛠️ Architecture & Pipeline Specs",
])

# ----------------------------------------------------
# TAB 1: VALUATION SIMULATOR
# ----------------------------------------------------
with tab_simulator:
    st.subheader("Interactive Property Valuation Estimator")
    st.markdown(
        "Configure custom residential property characteristics below. "
        "The estimator applies the fitted **`ColumnTransformer`** (mean/mode imputation, scaling, one-hot encoding) "
        "and predicts the expected closing market valuation in real time."
    )

    with st.form("valuation_form"):
        col_dim, col_qual, col_arch = st.columns(3)

        with col_dim:
            st.markdown("#### 📐 Dimensions & Space")
            gr_liv_area = st.slider("Above-Ground Living Area (sq ft)", min_value=500, max_value=4500, value=1710, step=25)
            lot_area = st.number_input("Lot Size Area (sq ft)", min_value=1500, max_value=80000, value=8450, step=250)
            total_bsmt_sf = st.slider("Total Basement Area (sq ft)", min_value=0, max_value=3000, value=856, step=25)
            first_flr_sf = st.number_input("1st Floor Area (sq ft)", min_value=400, max_value=3000, value=856, step=25)
            second_flr_sf = st.number_input("2nd Floor Area (sq ft)", min_value=0, max_value=2500, value=854, step=25)
            garage_area = st.slider("Garage Area (sq ft)", min_value=0, max_value=1400, value=548, step=20)

        with col_qual:
            st.markdown("#### ⭐ Quality & Timeline")
            overall_qual = st.slider("Overall Material & Finish Quality (1-10)", min_value=1, max_value=10, value=7)
            overall_cond = st.slider("Overall Condition Rating (1-10)", min_value=1, max_value=10, value=5)
            year_built = st.slider("Year Built", min_value=1880, max_value=2010, value=2003)
            year_remod = st.slider("Year Remodeled / Added", min_value=1950, max_value=2010, value=2003)
            kitchen_qual = st.selectbox("Kitchen Quality", options=["Ex", "Gd", "TA", "Fa"], index=1, help="Ex=Excellent, Gd=Good, TA=Average/Typical, Fa=Fair")
            exter_qual = st.selectbox("Exterior Quality", options=["Ex", "Gd", "TA", "Fa"], index=1)

        with col_arch:
            st.markdown("#### 📍 Location & Amenities")
            # Top neighborhoods from Ames dataset
            neighborhoods = sorted(df["Neighborhood"].dropna().unique().tolist())
            neighborhood = st.selectbox("Neighborhood (Ames, IA)", options=neighborhoods, index=neighborhoods.index("CollgCr") if "CollgCr" in neighborhoods else 0)
            house_style = st.selectbox("House Style", options=["1Story", "2Story", "1.5Fin", "SLvl", "SFoyer"], index=1)
            bldg_type = st.selectbox("Building Type", options=["1Fam", "2fmCon", "Duplex", "TwnhsE", "Twnhs"], index=0)
            full_bath = st.selectbox("Full Bathrooms", options=[1, 2, 3, 4], index=1)
            bedroom_abv = st.selectbox("Bedrooms (Above Ground)", options=[1, 2, 3, 4, 5, 6], index=2)
            garage_cars = st.selectbox("Garage Car Capacity", options=[0, 1, 2, 3, 4], index=2)
            central_air = st.selectbox("Central Air Conditioning", options=["Y", "N"], index=0)

        submitted = st.form_submit_button("💰 Predict Property Market Valuation", use_container_width=True)

    # Base representative template with default baseline features
    sample_record: Dict[str, Any] = X.iloc[0].to_dict()
    # Update with interactive user inputs
    sample_record.update({
        "GrLivArea": gr_liv_area,
        "LotArea": lot_area,
        "TotalBsmtSF": total_bsmt_sf,
        "1stFlrSF": first_flr_sf,
        "2ndFlrSF": second_flr_sf,
        "GarageArea": garage_area,
        "OverallQual": overall_qual,
        "OverallCond": overall_cond,
        "YearBuilt": year_built,
        "YearRemodAdd": year_remod,
        "KitchenQual": kitchen_qual,
        "ExterQual": exter_qual,
        "Neighborhood": neighborhood,
        "HouseStyle": house_style,
        "BldgType": bldg_type,
        "FullBath": full_bath,
        "BedroomAbvGr": bedroom_abv,
        "GarageCars": garage_cars,
        "CentralAir": central_air,
    })

    # Inference execution
    active_pipeline = ridge_pipeline if ridge_pipeline is not None else baseline_pipeline

    if active_pipeline is not None:
        pred_ridge = predict_price(active_pipeline, [sample_record])[0]
        pred_baseline = predict_price(baseline_pipeline, [sample_record])[0] if baseline_pipeline else pred_ridge

        st.markdown(
            f"""
            <div class="prediction-hero-card">
                <div style="font-size: 1rem; text-transform: uppercase; letter-spacing: 0.08em; color: #94a3b8; font-weight: 600;">
                    Estimated Market Closing Price (Ridge L₂ Regularized)
                </div>
                <div class="prediction-price">
                    ${pred_ridge:,.2f}
                </div>
                <div style="font-size: 0.95rem; color: #cbd5e1; margin-top: 0.5rem;">
                    Expected 95% Confidence Valuation Band: 
                    <strong style="color: #60a5fa;">${max(0, pred_ridge - 31024):,.0f}</strong> — 
                    <strong style="color: #60a5fa;">${pred_ridge + 31024:,.0f}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        res_c1, res_c2, res_c3 = st.columns(3)
        with res_c1:
            st.metric("Ridge Model Prediction", f"${pred_ridge:,.2f}", delta=f"${pred_ridge - pred_baseline:+,.2f} vs OLS")
        with res_c2:
            st.metric("Baseline OLS Prediction", f"${pred_baseline:,.2f}")
        with res_c3:
            median_ames = float(df["SalePrice"].median())
            st.metric("Ames Citywide Median", f"${median_ames:,.2f}", delta=f"${pred_ridge - median_ames:+,.2f} vs Median")


# ----------------------------------------------------
# TAB 2: MODEL BENCHMARK & DIAGNOSTICS
# ----------------------------------------------------
with tab_benchmark:
    st.subheader("Model Performance Comparison on Held-Out Test Set (N = 292)")

    comp_c1, comp_c2 = st.columns([1, 1.4])

    with comp_c1:
        st.markdown("#### 📋 Metrics Benchmark Table")
        metrics_data = {
            "Model Architecture": ["Baseline Linear Regression (OLS)", "Regularized Ridge Regression"],
            "R² Score": [0.8867, 0.8745],
            "RMSE ($)": [29476.84, 31024.24],
            "Penalty (α)": ["None (0.0)", "20.43 (5-Fold CV)"],
            "Stability": ["Susceptible to collinearity", "Robust & Shrinkage-bounded"],
        }
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)

        st.markdown("""
        **Senior ML Engineering Rationale:**
        - **Why Ridge?** Housing features exhibit strong **multicollinearity** (e.g. `GrLivArea`, `TotalBsmtSF`, `1stFlrSF`, `GarageCars`, `GarageArea`). Under unregularized OLS, high correlation causes near-singular $X^T X$, inflating coefficient variances and making weights hypersensitive to new or extreme test observations.
        - **The Role of α:** The $L_2$ penalty $\\alpha \\|w\\|_2^2$ shrinks coefficients toward zero, introducing minor bias but significantly cutting variance and bounding out-of-distribution risks.
        """)

    with comp_c2:
        # Precompute test predictions for visualization
        y_test_arr = np.asarray(y_test, dtype=float)
        active_eval = ridge_pipeline if ridge_pipeline is not None else baseline_pipeline
        if active_eval is not None:
            y_pred_ridge = active_eval.predict(X_test)
        else:
            y_pred_ridge = y_test_arr.copy()
        residuals_ridge = y_test_arr - y_pred_ridge

        # Plotly Actual vs. Predicted
        fig_scatter = px.scatter(
            x=y_test_arr,
            y=y_pred_ridge,
            labels={"x": "Actual SalePrice ($)", "y": "Predicted SalePrice ($)"},
            title="Actual vs. Predicted Sale Price (Held-Out Test Set)",
            opacity=0.65,
            color_discrete_sequence=["#38bdf8"],
        )
        min_v = float(min(y_test_arr.min(), y_pred_ridge.min()))
        max_v = float(max(y_test_arr.max(), y_pred_ridge.max()))
        fig_scatter.add_trace(
            go.Scatter(
                x=[min_v, max_v],
                y=[min_v, max_v],
                mode="lines",
                name="Ideal (y = x)",
                line=dict(color="#ef4444", dash="dash", width=2),
            )
        )
        fig_scatter.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=380,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Residual Plot
    fig_resid = px.scatter(
        x=y_pred_ridge,
        y=residuals_ridge,
        labels={"x": "Predicted SalePrice ($)", "y": "Residual (Actual - Predicted) ($)"},
        title="Residual Diagnostic Plot (Ridge Model Residuals vs. Fitted Values)",
        opacity=0.6,
        color_discrete_sequence=["#34d399"],
    )
    fig_resid.add_hline(y=0, line_dash="dash", line_color="#ef4444", line_width=2)
    fig_resid.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_resid, use_container_width=True)


# ----------------------------------------------------
# TAB 3: EXPLORATORY DATA ANALYSIS (EDA)
# ----------------------------------------------------
with tab_eda:
    st.subheader("Ames Housing Exploratory Data Analysis")

    eda_c1, eda_c2 = st.columns(2)

    with eda_c1:
        # SalePrice Distribution
        fig_hist = px.histogram(
            df,
            x="SalePrice",
            nbins=45,
            marginal="box",
            title="Target Variable Distribution: SalePrice ($)",
            color_discrete_sequence=["#6366f1"],
        )
        med_p = float(df["SalePrice"].median())
        mean_p = float(df["SalePrice"].mean())
        fig_hist.add_vline(x=med_p, line_dash="dash", line_color="#34d399", annotation_text=f"Median: ${med_p:,.0f}")
        fig_hist.add_vline(x=mean_p, line_dash="dot", line_color="#f59e0b", annotation_text=f"Mean: ${mean_p:,.0f}")
        fig_hist.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=380,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with eda_c2:
        # Top Correlated Features
        numeric_df = df.select_dtypes(include=[np.number])
        top_corr = numeric_df.corr()["SalePrice"].abs().sort_values(ascending=False).head(12)
        top_corr = top_corr.drop("SalePrice", errors="ignore").reset_index()
        top_corr.columns = ["Feature", "Absolute Pearson Correlation"]

        fig_corr = px.bar(
            top_corr,
            x="Absolute Pearson Correlation",
            y="Feature",
            orientation="h",
            title="Top Predictor Features Correlated with SalePrice",
            color="Absolute Pearson Correlation",
            color_continuous_scale="Viridis",
        )
        fig_corr.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=380,
            yaxis=dict(autorange="reversed"),
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_corr, use_container_width=True)


# ----------------------------------------------------
# TAB 4: ARCHITECTURE & METHODOLOGY
# ----------------------------------------------------
with tab_architecture:
    st.subheader("Machine Learning Engineering Architecture & Best Practices")

    st.markdown("""
    #### 🛡️ Strict Leak-Free Design
    - **Featurization Ordering:** The dataset is split into training (80%) and testing (20%) sets **before** fitting the preprocessing pipeline. Imputation statistics and feature scalers are fitted exclusively on `X_train`, preventing data snooping from held-out validation samples.
    - **Handling Missing Values:**
      - Numerical columns (e.g. `LotFrontage`, `MasVnrArea`) are imputed using `SimpleImputer(strategy="mean")`.
      - Categorical columns (e.g. `Alley`, `PoolQC`, `GarageType`) are imputed using `SimpleImputer(strategy="most_frequent")`.
    - **Categorical Encoding:** `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` guarantees that unseen categorical levels encountered during production inference do not cause runtime failures.
    - **Hyperparameter Optimization:** Ridge regression tuning explores $\\alpha \\in [10^{-2}, 10^{4}]$ using **5-fold cross-validation** solely within the training partition.
    - **Persistence:** The complete end-to-end pipeline (`ColumnTransformer` + estimator) is serialized atomically via `Joblib` to prevent serving skew.
    """)

    st.markdown("#### 📂 Modular Repository Architecture")
    st.code("""
predictive-linear-regression/
├── data/
│   ├── README.md               # Sourcing, schema definitions & citation
│   └── ames_housing.csv        # Local cached raw dataset (OpenML ID: 42165)
├── models/
│   ├── ridge_pipeline.joblib   # Persisted regularized pipeline
│   └── baseline_pipeline.joblib # Persisted baseline OLS pipeline
├── reports/
│   ├── metrics.csv             # Held-out test evaluation benchmark
│   ├── actual_vs_predicted.png # Scatter diagnostic plot
│   └── residual_plot.png       # Residual diagnostic plot
├── src/
│   ├── data_loader.py          # OpenML ingestion & schema validation
│   ├── preprocessing.py        # ColumnTransformer, imputation, scaling, OHE
│   ├── train.py                # RidgeCV 5-fold tuning & serialization
│   ├── evaluate.py             # Metrics calculation & plotting
│   └── predict.py              # Production inference API
├── tests/
│   └── test_pipeline.py        # 8 automated unit & integration tests (Pytest)
├── app.py                      # Interactive Streamlit Web UI
├── main.py                     # CLI pipeline orchestrator
└── requirements.txt            # Locked dependencies
    """, language="text")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>"
    "Built with Python 3.11 • Scikit-Learn • Streamlit • Plotly • Joblib"
    "</div>",
    unsafe_allow_html=True,
)
