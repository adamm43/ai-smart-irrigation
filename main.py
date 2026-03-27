"""Smart Irrigation AI — Full Pipeline CLI"""
 
import os, sys, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
 
from src.utils import setup_logging
setup_logging()
import logging
logger = logging.getLogger(__name__)
 
from src.data_preparation import load_and_clean_data
from src.regression_model import (
    train_linear_regression, train_ridge,
    train_random_forest, tune_random_forest,
    train_gradient_boosting, tune_gradient_boosting,
    evaluate_regression, cross_validate_model,
    get_feature_importance, save_model,
)
from src.neural_network import build_nn, train_nn, evaluate_nn, save_nn
from src.evaluation import compute_metrics, build_leaderboard
 
 
def main():
    DATA = "data/raw/smart_irrigation_4000.csv"
    logger.info("=" * 55)
    logger.info("💧 Smart Irrigation AI — Pipeline")
    logger.info("=" * 55)
 
    X_train, X_test, y_train, y_test, _, feats = load_and_clean_data(DATA)
    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")
 
    metrics = []
 
    for name, fn in [
        ("Linear Regression",  lambda: train_linear_regression(X_train, y_train)),
        ("Ridge Regression",   lambda: train_ridge(X_train, y_train)),
    ]:
        logger.info(f"— {name}")
        m = fn()
        r = evaluate_regression(m, X_test, y_test)
        metrics.append(compute_metrics(y_test, r["y_pred"], name))
        save_model(m, name.lower().replace(" ", "_"))
 
    logger.info("— Random Forest (tuned)")
    rf = tune_random_forest(X_train, y_train, cv=5, n_iter=20)
    r  = evaluate_regression(rf, X_test, y_test)
    metrics.append(compute_metrics(y_test, r["y_pred"], "Random Forest"))
    save_model(rf, "random_forest")
 
    logger.info("— Gradient Boosting (tuned)")
    gbm = tune_gradient_boosting(X_train, y_train, cv=5, n_iter=20)
    r   = evaluate_regression(gbm, X_test, y_test)
    metrics.append(compute_metrics(y_test, r["y_pred"], "Gradient Boosting"))
    save_model(gbm, "gradient_boosting")
 
    logger.info("— Neural Network")
    nn = build_nn(input_dim=X_train.shape[1])
    nn, _ = train_nn(nn, X_train, y_train, X_val=X_test, y_val=y_test, epochs=200)
    r  = evaluate_nn(nn, X_test, y_test)
    metrics.append(compute_metrics(y_test, r["y_pred"], "Neural Network"))
    save_nn(nn, "models/neural_network.keras")
 
    logger.info("=" * 55)
    lb = build_leaderboard(metrics)
    print("\n" + lb.to_string(index=False))
    best = lb.iloc[0]
    logger.info(f"🏆 Best: {best['Model']} | R²:{best['R²']} | RMSE:{best['RMSE']}L")
    logger.info("Run: python3 -m streamlit run app_dashboard.py")
 
 
if __name__ == "__main__":
    main()