import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

ACCENT = "#2196F3"
WARN   = "#FF5722"
GREEN  = "#4CAF50"


def _style(ax, title="", xlabel="", ylabel=""):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


# ─── EDA ─────────────────────────────────────────────────────────────────────

def plot_distribution(df: pd.DataFrame, col: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df[col].dropna(), kde=True, color=ACCENT, ax=ax, bins=30)
    _style(ax, f"Distribution of {col}", col, "Count")
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    corr = df.select_dtypes(include=np.number).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(max(8, len(corr)*0.9), max(6, len(corr)*0.8)))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax, annot_kws={"size": 9})
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_target_vs_feature(df: pd.DataFrame, feature: str, target: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    valid = df[[feature, target]].dropna()
    ax.scatter(valid[feature], valid[target], alpha=0.3, color=ACCENT, s=15)
    if len(valid) > 1:
        m, b, *_ = stats.linregress(valid[feature], valid[target])
        x_ = np.linspace(valid[feature].min(), valid[feature].max(), 100)
        r  = np.corrcoef(valid[feature], valid[target])[0, 1]
        ax.plot(x_, m*x_+b, color=WARN, linewidth=2, label=f"r = {r:.2f}")
        ax.legend()
    _style(ax, f"{feature} vs {target}", feature, target)
    plt.tight_layout()
    return fig


def plot_missing_values(df: pd.DataFrame) -> plt.Figure:
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(5, 2))
    if missing.empty:
        ax.text(0.5, 0.5, "✅ No missing values!", ha="center", va="center",
                fontsize=14, color="green")
        ax.axis("off")
    else:
        missing.plot(kind="bar", color=WARN, ax=ax)
        _style(ax, "Missing Values", "Column", "Count")
    plt.tight_layout()
    return fig


def plot_boxplots(df: pd.DataFrame) -> plt.Figure:
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    cols = 4
    rows = max(1, (len(numeric) + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.5, rows*3))
    axes = axes.flatten() if rows*cols > 1 else [axes]
    for i, col in enumerate(numeric):
        axes[i].boxplot(df[col].dropna(), patch_artist=True,
                        boxprops=dict(facecolor=ACCENT, alpha=0.6))
        axes[i].set_title(col, fontsize=9)
        axes[i].spines["top"].set_visible(False)
        axes[i].spines["right"].set_visible(False)
    for j in range(len(numeric), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Boxplots — Outlier Detection", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_target_by_category(df: pd.DataFrame, cat_col: str, target: str) -> plt.Figure:
    """Boxplot of target grouped by a categorical feature."""
    fig, ax = plt.subplots(figsize=(8, 4))
    groups = [df[df[cat_col] == v][target].dropna().values
               for v in df[cat_col].unique()]
    labels = df[cat_col].unique().tolist()
    bp = ax.boxplot(groups, patch_artist=True, labels=labels,
                    boxprops=dict(facecolor=ACCENT, alpha=0.7))
    _style(ax, f"{target} by {cat_col}", cat_col, target)
    plt.xticks(rotation=20)
    plt.tight_layout()
    return fig


# ─── Model Evaluation ────────────────────────────────────────────────────────

def plot_actual_vs_predicted(y_true, y_pred, model_name="Model") -> plt.Figure:
    from sklearn.metrics import r2_score
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, color=ACCENT, s=20)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=2, label="Perfect fit")
    r2 = r2_score(y_true, y_pred)
    ax.annotate(f"R² = {r2:.3f}", xy=(0.05, 0.92), xycoords="axes fraction",
                fontsize=12, bbox=dict(boxstyle="round", facecolor="lightyellow"))
    _style(ax, f"{model_name} — Actual vs Predicted", "Actual (L)", "Predicted (L)")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_residuals(y_true, y_pred, model_name="Model") -> plt.Figure:
    y_true    = np.array(y_true).flatten()
    y_pred    = np.array(y_pred).flatten()
    residuals = y_true - y_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.scatter(y_pred, residuals, alpha=0.4, color=ACCENT, s=15)
    ax1.axhline(0, color=WARN, linewidth=1.5, linestyle="--")
    _style(ax1, "Residuals vs Fitted", "Fitted Values", "Residuals")
    (osm, osr), (slope, intercept, _) = stats.probplot(residuals, dist="norm")
    ax2.scatter(osm, osr, alpha=0.4, color=ACCENT, s=15)
    ax2.plot(osm, slope*np.array(osm)+intercept, color=WARN, linewidth=2)
    _style(ax2, "Q-Q Plot", "Theoretical Quantiles", "Sample Quantiles")
    plt.suptitle(f"{model_name} — Residual Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_feature_importance(df_imp: pd.DataFrame, top_n=15) -> plt.Figure:
    df = df_imp.head(top_n).copy()
    fig, ax = plt.subplots(figsize=(8, max(4, top_n*0.45)))
    bars = ax.barh(df["feature"][::-1], df["importance"][::-1],
                   color=ACCENT, alpha=0.85)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    _style(ax, f"Top {top_n} Feature Importances", "Importance", "Feature")
    plt.tight_layout()
    return fig


def plot_model_comparison(leaderboard: pd.DataFrame) -> plt.Figure:
    models = leaderboard["Model"].tolist()
    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    items = [("MSE", WARN), ("RMSE", "#FF9800"), ("MAE", "#FFC107"), ("R²", GREEN)]
    for i, (metric, color) in enumerate(items):
        vals = leaderboard[metric].tolist()
        bars = axes[i].bar(x, vals, color=color, alpha=0.85, edgecolor="white")
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(models, rotation=20, ha="right", fontsize=9)
        axes[i].bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
        _style(axes[i], metric, "", metric)
    plt.suptitle("Model Comparison — All Metrics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_nn_history(history) -> plt.Figure:
    h = history.history
    epochs = range(1, len(h["loss"]) + 1)
    has_lr = "lr" in h
    n = 3 if has_lr else 2
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    axes[0].plot(epochs, h["loss"], label="Train", color=ACCENT)
    if "val_loss" in h:
        axes[0].plot(epochs, h["val_loss"], label="Val", color=WARN)
    _style(axes[0], "Loss (Huber)", "Epoch", "Loss")
    axes[0].legend()
    if "mae" in h:
        axes[1].plot(epochs, h["mae"], label="Train", color=ACCENT)
        if "val_mae" in h:
            axes[1].plot(epochs, h["val_mae"], label="Val", color=WARN)
        _style(axes[1], "MAE", "Epoch", "MAE")
        axes[1].legend()
    if has_lr and n == 3:
        axes[2].plot(epochs, h["lr"], color="#9C27B0")
        axes[2].set_yscale("log")
        _style(axes[2], "Learning Rate", "Epoch", "LR")
    plt.suptitle("Neural Network Training History", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_prediction_vs_index(y_true, y_pred, n=100, model_name="Model") -> plt.Figure:
    n = min(n, len(y_true))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(range(n), np.array(y_true)[:n], label="Actual",
            color=ACCENT, linewidth=1.5)
    ax.plot(range(n), np.array(y_pred)[:n], label="Predicted",
            color=WARN, linewidth=1.5, linestyle="--")
    _style(ax, f"{model_name} — Prediction vs Actual (first {n} samples)",
           "Sample Index", "Water Amount (L)")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_irrigation_rules(df: pd.DataFrame, target: str = "water_amount_liters") -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if "crop_type" in df.columns:
        avg_crop = df.groupby("crop_type")[target].mean().sort_values(ascending=False)
        bars = axes[0].bar(avg_crop.index, avg_crop.values, color=ACCENT, alpha=0.85)
        axes[0].bar_label(bars, fmt="%.0f L", padding=3, fontsize=10)
        _style(axes[0], "Avg Water Need by Crop Type", "Crop", "Avg Litres/day")

    if "season" in df.columns:
        avg_season = df.groupby("season")[target].mean().sort_values(ascending=False)
        bars = axes[1].bar(avg_season.index, avg_season.values, color=GREEN, alpha=0.85)
        axes[1].bar_label(bars, fmt="%.0f L", padding=3, fontsize=10)
        _style(axes[1], "Avg Water Need by Season", "Season", "Avg Litres/day")

    plt.suptitle("💧 Irrigation Patterns — Domain Insights",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig