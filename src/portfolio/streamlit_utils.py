# src/portfolio/streamlit_utils.py
from __future__ import annotations
import os
import pandas as pd
from datetime import datetime
from typing import Tuple, Optional, List

DATA_DIR = os.path.join("Data", "processed")   # <-- matches your screenshot

def _csv_path_for(ticker: str) -> str:
    fname = f"{ticker.upper()}_clean.csv"
    return os.path.join(DATA_DIR, fname)

def load_price_df(ticker: str) -> pd.DataFrame:
    """
    Load a processed CSV for a ticker. Returns a DataFrame indexed by Date.
    Works if Date is either a column or already the index.
    """
    path = _csv_path_for(ticker)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected file not found: {path}")
    df = pd.read_csv(path)
    # Normalize date index
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    else:
        df.index = pd.to_datetime(df.index)
    # Keep common columns if present
    cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Ticker"] if c in df.columns]
    return df[cols].sort_index()

#def filter_by_date(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    return df.loc[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))]

def filter_by_date(df: pd.DataFrame, start, end) -> pd.DataFrame:
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    return df.loc[(df.index >= start) & (df.index <= end)]


def latest_price_and_change(df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (last_close, pct_change_vs_prev_close) in decimal units (e.g., 0.012 = 1.2%).
    """
    if "Close" not in df.columns or len(df) < 2:
        return None, None
    last = df["Close"].iloc[-1]
    prev = df["Close"].iloc[-2]
    pct = (last - prev) / prev if prev != 0 else None
    return float(last), (float(pct) if pct is not None else None)

def last_updated_timestamp(paths: Optional[List[str]] = None) -> str:
    """
    Find the most recent modified time among common artifact folders.
    Returns a nice UTC string for display.
    """
    paths = paths or [
        os.path.join("Data", "processed"),
        os.path.join("results", "forecasts"),
        os.path.join("models", "saved"),
    ]
    mtimes = []
    for p in paths:
        if not os.path.exists(p):
            continue
        for root, _, files in os.walk(p):
            for f in files:
                fp = os.path.join(root, f)
                try:
                    mtimes.append(os.path.getmtime(fp))
                except OSError:
                    pass
    if not mtimes:
        return "N/A"
    dt = datetime.utcfromtimestamp(max(mtimes))
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")



# --- LSTM forecast helpers (overlay-ready) ---
import numpy as np
from typing import Iterable, Dict, Any

def to_trading_days(months: int) -> int:
    return int(round(months * 21))

def get_last_close_window(ts: pd.Series, lookback: int) -> np.ndarray:
    """
    Return the last 'lookback' closes as shape (lookback, 1) for the LSTM.
    """
    vals = ts.dropna().values.astype("float32")
    if len(vals) < lookback:
        raise ValueError(f"Series shorter than lookback={lookback}")
    window = vals[-lookback:]
    return window.reshape(-1, 1)

def annualized_return_from_path(forecast_close: Iterable[float]) -> float:
    """
    Simple annualized return from first to last forecasted close.
    """
    fc = list(forecast_close)
    if len(fc) < 2 or fc[0] == 0:
        return np.nan
    total_r = (fc[-1] / fc[0]) - 1.0
    # assume path is for N trading days; annualize with 252
    n = len(fc)
    return (1.0 + total_r) ** (252.0 / n) - 1.0

def build_ci_band(series: pd.Series, pct: float = 0.05) -> Dict[str, Any]:
    """
    Build a ±pct band around a forecast series (rough proxy CI).
    Returns dict with 'lower' and 'upper' pd.Series.
    """
    lower = series * (1.0 - pct)
    upper = series * (1.0 + pct)
    return {"lower": lower, "upper": upper}
