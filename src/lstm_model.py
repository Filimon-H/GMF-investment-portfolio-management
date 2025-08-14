# src/lstm_model.py
from __future__ import annotations

import os
from typing import Tuple, Optional

import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

try:
    import joblib  # for loading the saved scaler (recommended)
except Exception:
    joblib = None

# -----------------------
# Paths to saved artifacts
# -----------------------
MODEL_PATH = os.path.join("models", "saved", "tsla_lstm_model.keras")
SCALER_PATH = os.path.join("models", "saved", "tsla_scaler.pkl")  # optional but recommended

# -----------------------
# Training-time utilities
# -----------------------
def create_lstm_sequences(data: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a 1D array into LSTM-ready (X, y) sequences.
    X shape: (num_samples, window_size, 1)
    y shape: (num_samples, 1)
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size])
    X = np.array(X).reshape(-1, window_size, 1)
    y = np.array(y).reshape(-1, 1)
    return X, y


def build_lstm_model(input_shape: Tuple[int, int]) -> Sequential:
    """
    Build a simple LSTM model using Keras Sequential API.
    input_shape should be (window_size, num_features) e.g., (60, 1)
    """
    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

# -----------------------
# Inference-time helpers (EXPOSED)
# -----------------------
def load_lstm_and_scaler() -> Tuple:
    """
    Load the saved Keras model (.keras) and the MinMaxScaler (if present).
    Returns (model, scaler_or_None)
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Saved LSTM model not found at: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    scaler = None
    if joblib is not None and os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
        except Exception as e:
            # We can still proceed without a scaler, but it's better to log/raise in your app if needed
            scaler = None

    return model, scaler


def get_last_close_window(close_series, lookback: int) -> np.ndarray:
    """
    Build the last lookback-length window from a Pandas Series of Close prices.
    Returns shape (lookback, 1) on the raw (unscaled) price level.
    """
    values = np.asarray(close_series.dropna(), dtype="float32")
    if len(values) < lookback:
        raise ValueError(f"Series length {len(values)} is shorter than lookback={lookback}")
    window = values[-lookback:]
    return window.reshape(-1, 1)


def forecast_n_steps_from_window(
    last_window: np.ndarray,
    n_steps: int,
    model: Optional[Sequential] = None,
    scaler: Optional[MinMaxScaler] = None,
) -> np.ndarray:
    """
    Iteratively forecast n_steps ahead given the last 'lookback' window of Close prices.

    Parameters
    ----------
    last_window : np.ndarray
        Shape (lookback, 1), raw Close values (unscaled).
    n_steps : int
        Number of future trading days to forecast.
    model : keras.Model, optional
        If None, loads from MODEL_PATH.
    scaler : MinMaxScaler, optional
        If provided, will scale/inverse_scale around the predictions.

    Returns
    -------
    np.ndarray
        Forecasted Close prices on the original (unscaled) level, shape (n_steps,).
    """
    if model is None:
        model, _maybe_scaler = load_lstm_and_scaler()
        scaler = scaler or _maybe_scaler  # prefer explicitly passed scaler if provided

    # Prepare the input window (scale if we can)
    seq = last_window.astype("float32")
    use_scaler = scaler is not None
    if use_scaler:
        seq_scaled = scaler.transform(seq)
    else:
        seq_scaled = seq

    # Model expects (1, lookback, 1)
    x = seq_scaled.reshape(1, seq_scaled.shape[0], 1)
    preds_scaled = []

    for _ in range(int(n_steps)):
        yhat = model.predict(x, verbose=0)  # shape (1,1)
        preds_scaled.append(yhat[0, 0])
        # Roll the window forward by one and append yhat
        x = np.concatenate([x[:, 1:, :], yhat.reshape(1, 1, 1)], axis=1)

    preds_scaled = np.array(preds_scaled).reshape(-1, 1)

    # Invert scale if necessary
    if use_scaler:
        preds = scaler.inverse_transform(preds_scaled)
    else:
        preds = preds_scaled

    return preds.flatten()
