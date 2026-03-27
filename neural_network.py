import numpy as np
import logging
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_nn(input_dim: int, layer1=128, layer2=64, layer3=32,
             dropout_rate=0.3, learning_rate=0.001, l2_reg=1e-4) -> keras.Model:
    set_seed()
    model = Sequential([
        Input(shape=(input_dim,)),

        Dense(layer1, kernel_initializer="he_normal", kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        keras.layers.Activation("relu"),
        Dropout(dropout_rate),

        Dense(layer2, kernel_initializer="he_normal", kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        keras.layers.Activation("relu"),
        Dropout(dropout_rate),

        Dense(layer3, kernel_initializer="he_normal", kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        keras.layers.Activation("relu"),
        Dropout(dropout_rate / 2),

        Dense(1, activation="linear"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="huber",
        metrics=["mae", "mse"],
    )
    logger.info(f"NN: {layer1}→{layer2}→{layer3}→1 | params: {model.count_params():,}")
    return model


def train_nn(model, X_train, y_train, X_val=None, y_val=None,
             epochs=200, batch_size=32) -> tuple:
    """Train with EarlyStopping + ReduceLROnPlateau."""
    monitor = "val_loss" if X_val is not None else "loss"
    callbacks = [
        EarlyStopping(monitor=monitor, patience=20,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor=monitor, factor=0.5,
                          patience=8, min_lr=1e-6, verbose=0),
    ]
    fit_kwargs = dict(x=X_train, y=y_train,
                      epochs=epochs, batch_size=batch_size,
                      callbacks=callbacks, verbose=0)
    if X_val is not None:
        fit_kwargs["validation_data"] = (X_val, y_val)

    history = model.fit(**fit_kwargs)
    logger.info(f"NN done — {len(history.history['loss'])} epochs.")
    return model, history


def evaluate_nn(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test, verbose=0).flatten()
    y_true = np.array(y_test).flatten()
    return {
        "RMSE":   np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE":    mean_absolute_error(y_true, y_pred),
        "R2":     r2_score(y_true, y_pred),
        "y_pred": y_pred,
        "y_test": y_true,
    }


def save_nn(model, path: str = "models/neural_network.keras"):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    model.save(path)
    logger.info(f"NN saved → {path}")


def load_nn(path: str = "models/neural_network.keras"):
    from keras.models import load_model as keras_load
    return keras_load(path)