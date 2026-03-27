import os, logging, joblib
import pandas as pd

def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level,
                        format="%(asctime)s — %(levelname)s — %(message)s",
                        datefmt="%H:%M:%S")

def save_model(model, name: str, models_dir: str = "models") -> str:
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, f"{name}.pkl")
    joblib.dump(model, path)
    return path

def load_model(name: str, models_dir: str = "models"):
    path = os.path.join(models_dir, f"{name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model at {path}")
    return joblib.load(path)

def history_to_dataframe(history) -> pd.DataFrame:
    return pd.DataFrame(history.history)