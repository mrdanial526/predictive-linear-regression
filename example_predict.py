"""Example script demonstrating inference with the persisted Ridge pipeline."""

from src.predict import load_model, predict_price


def main():
    # 1. Load persisted pipeline
    model_path = "models/ridge_pipeline.joblib"
    print(f"Loading trained pipeline from: {model_path}")
    model = load_model(model_path)

    # 2. Define sample property features
    sample_property = {
        "MSSubClass": 60,
        "MSZoning": "RL",
        "LotFrontage": 65.0,
        "LotArea": 8450,
        "Street": "Pave",
        "LotShape": "Reg",
        "LandContour": "Lvl",
        "Utilities": "AllPub",
        "LotConfig": "Inside",
        "LandSlope": "Gtl",
        "Neighborhood": "CollgCr",
        "Condition1": "Norm",
        "BldgType": "1Fam",
        "HouseStyle": "2Story",
        "OverallQual": 7,
        "OverallCond": 5,
        "YearBuilt": 2003,
        "YearRemodAdd": 2003,
        "RoofStyle": "Gable",
        "RoofMatl": "CompShg",
        "Exterior1st": "VinylSd",
        "Exterior2nd": "VinylSd",
        "MasVnrType": "BrkFace",
        "MasVnrArea": 196.0,
        "ExterQual": "Gd",
        "ExterCond": "TA",
        "Foundation": "PConc",
        "BsmtQual": "Gd",
        "BsmtCond": "TA",
        "BsmtExposure": "No",
        "BsmtFinType1": "GLQ",
        "BsmtFinSF1": 706,
        "BsmtFinType2": "Unf",
        "BsmtFinSF2": 0,
        "BsmtUnfSF": 150,
        "TotalBsmtSF": 856,
        "Heating": "GasA",
        "HeatingQC": "Ex",
        "CentralAir": "Y",
        "Electrical": "SBrkr",
        "1stFlrSF": 856,
        "2ndFlrSF": 854,
        "LowQualFinSF": 0,
        "GrLivArea": 1710,
        "BsmtFullBath": 1,
        "BsmtHalfBath": 0,
        "FullBath": 2,
        "HalfBath": 1,
        "BedroomAbvGr": 3,
        "KitchenAbvGr": 1,
        "KitchenQual": "Gd",
        "TotRmsAbvGrd": 8,
        "Functional": "Typ",
        "Fireplaces": 0,
        "GarageType": "Attchd",
        "GarageYrBlt": 2003.0,
        "GarageFinish": "RFn",
        "GarageCars": 2,
        "GarageArea": 548,
        "GarageQual": "TA",
        "GarageCond": "TA",
        "PavedDrive": "Y",
        "WoodDeckSF": 0,
        "OpenPorchSF": 61,
        "EnclosedPorch": 0,
        "3SsnPorch": 0,
        "ScreenPorch": 0,
        "PoolArea": 0,
        "MiscVal": 0,
        "MoSold": 2,
        "YrSold": 2008,
        "Alley": None,
        "Condition2": "Norm",
        "FireplaceQu": None,
        "PoolQC": None,
        "Fence": None,
        "MiscFeature": None,
        "SaleType": "WD",
        "SaleCondition": "Normal",
    }

    # 3. Predict valuation
    print("Predicting valuation for sample property...")
    valuation = predict_price(model, [sample_property])
    print(f"\n==================================================")
    print(f" Estimated Market Valuation: ${valuation[0]:,.2f}")
    print(f"==================================================")


if __name__ == "__main__":
    main()
