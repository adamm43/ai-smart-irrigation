import os

BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_PATH   = os.path.join(BASE_DIR, "data", "raw", "smart_irrigation_4000.csv")
MODELS_DIR      = os.path.join(BASE_DIR, "models")

TARGET_COLUMN   = "water_amount_liters"
TEST_SIZE       = 0.2
RANDOM_STATE    = 42
CV_FOLDS        = 5

COLS_TO_DROP = []

NN_CONFIG = {
    "layer1":        128,
    "layer2":        64,
    "layer3":        32,
    "dropout_rate":  0.3,
    "learning_rate": 0.001,
    "epochs":        200,
    "batch_size":    32,
    "patience":      20,
}