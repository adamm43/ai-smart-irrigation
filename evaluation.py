import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_metrics(y_true, y_pred, model_name: str = "Model") -> dict:
    """Compute MSE, RMSE, MAE, R², MAPE for a model."""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan
    return {
        "Model": model_name,
        "MSE":   round(mse,  2),
        "RMSE":  round(rmse, 2),
        "MAE":   round(mae,  2),
        "R²":    round(r2,   4),
        "MAPE%": round(mape, 2),
    }


def build_leaderboard(results: list) -> pd.DataFrame:
    """Build sorted leaderboard from list of compute_metrics() dicts."""
    df = pd.DataFrame(results).sort_values("R²", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df


def residual_analysis(y_true, y_pred) -> pd.DataFrame:
    y_true    = np.array(y_true).flatten()
    y_pred    = np.array(y_pred).flatten()
    residuals = y_true - y_pred
    return pd.DataFrame({
        "actual":       y_true,
        "predicted":    y_pred,
        "residual":     residuals,
        "abs_residual": np.abs(residuals),
        "pct_error":    np.where(y_true != 0,
                                 np.abs(residuals / y_true) * 100, np.nan),
    })