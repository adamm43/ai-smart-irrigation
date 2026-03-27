import numpy as np
import pandas as pd
import logging
import warnings
import os
import joblib

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ─── Trainers ────────────────────────────────────────────────────────────────

def train_linear_regression(X_train, y_train) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    logger.info("Linear Regression trained.")
    return model


def train_ridge(X_train, y_train, alpha: float = 1.0) -> Ridge:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    logger.info(f"Ridge trained (alpha={alpha}).")
    return model


def train_random_forest(X_train, y_train, n_estimators=100,
                        max_depth=None, random_state=42) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=random_state, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    logger.info(f"Random Forest trained ({n_estimators} trees).")
    return model


def train_gradient_boosting(X_train, y_train, n_estimators=200,
                             learning_rate=0.1, max_depth=4,
                             random_state=42) -> GradientBoostingRegressor:
    model = GradientBoostingRegressor(
        n_estimators=n_estimators, learning_rate=learning_rate,
        max_depth=max_depth, random_state=random_state,
        subsample=0.8, validation_fraction=0.1, n_iter_no_change=15,
    )
    model.fit(X_train, y_train)
    logger.info("Gradient Boosting trained.")
    return model


# ─── Hyperparameter Tuning ───────────────────────────────────────────────────

def tune_random_forest(X_train, y_train, cv=5, n_iter=20,
                       random_state=42) -> RandomForestRegressor:
    param_dist = {
        "n_estimators":      [100, 200, 300, 500],
        "max_depth":         [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
        "max_features":      ["sqrt", "log2"],
    }
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=random_state, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=n_iter, cv=cv,
        scoring="neg_root_mean_squared_error",
        random_state=random_state, n_jobs=-1, verbose=0,
    )
    search.fit(X_train, y_train)
    logger.info(f"RF best: {search.best_params_} | RMSE: {-search.best_score_:.2f}")
    return search.best_estimator_


def tune_gradient_boosting(X_train, y_train, cv=5, n_iter=20,
                            random_state=42) -> GradientBoostingRegressor:
    param_dist = {
        "n_estimators":     [100, 200, 300],
        "learning_rate":    [0.01, 0.05, 0.1, 0.2],
        "max_depth":        [3, 4, 5, 6],
        "subsample":        [0.7, 0.8, 1.0],
        "min_samples_leaf": [1, 2, 4],
    }
    search = RandomizedSearchCV(
        GradientBoostingRegressor(random_state=random_state),
        param_distributions=param_dist,
        n_iter=n_iter, cv=cv,
        scoring="neg_root_mean_squared_error",
        random_state=random_state, n_jobs=-1, verbose=0,
    )
    search.fit(X_train, y_train)
    logger.info(f"GBM best: {search.best_params_} | RMSE: {-search.best_score_:.2f}")
    return search.best_estimator_


# ─── Cross-Validation ────────────────────────────────────────────────────────

def cross_validate_model(model, X, y, cv=5) -> dict:
    rmse = -cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
    mae  = -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error",     n_jobs=-1)
    r2   =  cross_val_score(model, X, y, cv=cv, scoring="r2",                          n_jobs=-1)
    return {
        "CV_RMSE_mean": rmse.mean(), "CV_RMSE_std": rmse.std(),
        "CV_MAE_mean":  mae.mean(),  "CV_MAE_std":  mae.std(),
        "CV_R2_mean":   r2.mean(),   "CV_R2_std":   r2.std(),
    }


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_regression(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    return {
        "RMSE":   np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE":    mean_absolute_error(y_test, y_pred),
        "R2":     r2_score(y_test, y_pred),
        "y_pred": y_pred,
        "y_test": np.array(y_test),
    }


# ─── Feature Importance ──────────────────────────────────────────────────────

def get_feature_importance(model, feature_names,
                            X_test=None, y_test=None) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif X_test is not None and y_test is not None:
        perm = permutation_importance(model, X_test, y_test,
                                      n_repeats=10, random_state=42)
        importances = perm.importances_mean
    else:
        return pd.DataFrame()
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


# ─── Persistence ─────────────────────────────────────────────────────────────

def save_model(model, name: str, models_dir: str = "models") -> str:
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, f"{name}.pkl")
    joblib.dump(model, path)
    logger.info(f"Saved → {path}")
    return path


def load_model(name: str, models_dir: str = "models"):
    path = os.path.join(models_dir, f"{name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model at {path}")
    return joblib.load(path)