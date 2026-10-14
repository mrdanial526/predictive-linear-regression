# Ames Housing Dataset

## 1. Overview and Source
The **Ames Housing Dataset** was compiled by Dean De Cock (Truman State University) as an modernized alternative to the classic Boston Housing dataset. It provides a detailed, high-dimensional collection of residential home sales in Ames, Iowa, from 2006 to 2010.

- **Primary Source:** [OpenML](https://www.openml.org/) (Dataset name: `house_prices`, OpenML Data ID: `42165`).
- **Citation:** De Cock, D. (2011). *Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project*. Journal of Statistics Education, 19(3).
- **Scope:** 1,460 observations and 80 features (79 predictor features and 1 continuous target variable).

## 2. Target Variable
- **Target Feature:** `SalePrice`
- **Type:** Continuous numerical variable (positive integer/float).
- **Unit:** US Dollars ($).
- **Domain Distribution:** Typically right-skewed with a median sale price around $163,000 and standard deviation reflecting variance in residential property values.

## 3. Feature Breakdown
The dataset contains a rich combination of structural, geographic, condition, and quality attributes:

### Numerical Features (36 features)
- **Continuous Dimensions:** `LotFrontage`, `LotArea`, `MasVnrArea`, `BsmtFinSF1`, `BsmtFinSF2`, `BsmtUnfSF`, `TotalBsmtSF`, `1stFlrSF`, `2ndFlrSF`, `LowQualFinSF`, `GrLivArea`, `GarageArea`, `WoodDeckSF`, `OpenPorchSF`, `EnclosedPorch`, `3SsnPorch`, `ScreenPorch`, `PoolArea`, `MiscVal`.
- **Discrete & Temporal Counts:** `OverallQual`, `OverallCond`, `YearBuilt`, `YearRemodAdd`, `BsmtFullBath`, `BsmtHalfBath`, `FullBath`, `HalfBath`, `BedroomAbvGr`, `KitchenAbvGr`, `TotRmsAbvGrd`, `Fireplaces`, `GarageYrBlt`, `GarageCars`, `MoSold`, `YrSold`.

### Categorical Features (43 features)
- **Nominal:** `MSZoning`, `Street`, `Alley`, `LandContour`, `LotConfig`, `Neighborhood`, `Condition1`, `Condition2`, `BldgType`, `HouseStyle`, `RoofStyle`, `RoofMatl`, `Exterior1st`, `Exterior2nd`, `MasVnrType`, `Foundation`, `Heating`, `CentralAir`, `Electrical`, `GarageType`, `MiscFeature`, `SaleType`, `SaleCondition`.
- **Ordinal / Rating Categories:** `LotShape`, `Utilities`, `LandSlope`, `ExterQual`, `ExterCond`, `BsmtQual`, `BsmtCond`, `BsmtExposure`, `BsmtFinType1`, `BsmtFinType2`, `HeatingQC`, `KitchenQual`, `Functional`, `FireplaceQu`, `GarageFinish`, `GarageQual`, `GarageCond`, `PavedDrive`, `PoolQC`, `Fence`.

## 4. Ingestion and Caching Procedure
1. The ingestion pipeline (`src/data_loader.py`) attempts to fetch the dataset from OpenML using `sklearn.datasets.fetch_openml`.
2. To guarantee zero-network reproducibility and offline testing, the raw dataset is cached locally at `data/ames_housing.csv`.
3. If an existing `data/ames_housing.csv` is found, the loader reads directly from disk.
4. Identifiers such as `Id` are identified and excluded from the predictive feature matrix `X`.
5. Strict schema validation ensures the target variable `SalePrice` is present and non-null before splitting.
