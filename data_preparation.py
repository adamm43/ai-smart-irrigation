import pandas as pd
import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

TARGET_COLUMN = "water_amount_liters"
COLS_TO_DROP  = []  


def validate_dataset(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Dataset is empty.")
    if df.shape[0] < 50:
        raise ValueError(f"Dataset too small: {df.shape[0]} rows.")
    logger.info(f"Dataset validated: {df.shape[0]} rows × {df.shape[1]} columns.")


def detect_target_column(df: pd.DataFrame, candidates: list = None) -> str:
    if candidates is None:
        candidates = [
            "water_amount_liters", "water_amount", "irrigation_amount",
            "water_needed", "soil_moisture_%", "moisture",
        ]
    for col in candidates:
        if col in df.columns:
            logger.info(f"Target detected: '{col}'")
            return col
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        fallback = numeric_cols[-1]
        logger.warning(f"Fallback target: '{fallback}'")
        return fallback
    raise ValueError("Cannot detect target column.")


def load_and_clean_data(
    path: str,
    target_col: str = None,
    test_size: float = 0.2,
    random_state: int = 42,
):
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    logger.info(f"Loaded: {df.shape}")
    validate_dataset(df)

    if target_col is None:
        target_col = detect_target_column(df)
    if target_col not in df.columns:
        raise ValueError(f"Target '{target_col}' not in columns: {list(df.columns)}")

    # Drop useless cols
    cols_to_drop = [c for c in COLS_TO_DROP if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info(f"Dropped: {cols_to_drop}")

    # Clean
    before = len(df)
    df = df.drop_duplicates().dropna(subset=[target_col])
    logger.info(f"Cleaned: {before - len(df)} rows removed.")

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(float)

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols     = X.select_dtypes(include=np.number).columns.tolist()
    logger.info(f"Numeric ({len(numeric_cols)}): {numeric_cols}")
    logger.info(f"Categorical ({len(categorical_cols)}): {categorical_cols}")

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")), # Replace missing values with median
        ("scaler",  StandardScaler()), # drna normalize numeric features to have mean=0 and std=1
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")), # Replace missing values with the most frequent category
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)), # Convert categorical features to one-hot encoding
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline,     numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ], remainder="drop")

    # Split FIRST — fit preprocessor only on train
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    X_train_arr = preprocessor.fit_transform(X_train_raw)
    X_test_arr  = preprocessor.transform(X_test_raw)

    # Feature names
    try:
        cat_names = (
            preprocessor.named_transformers_["cat"]["encoder"]
            .get_feature_names_out(categorical_cols).tolist()
            if categorical_cols else []
        )
    except Exception:
        cat_names = []
    feature_names = numeric_cols + cat_names

    #convert back to DataFrame
    X_train = pd.DataFrame(X_train_arr, columns=feature_names)
    X_test  = pd.DataFrame(X_test_arr,  columns=feature_names)
    y_train = y_train.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)

    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, preprocessor, feature_names


def get_raw_dataframe(path: str) -> pd.DataFrame:
    return pd.read_csv(path)