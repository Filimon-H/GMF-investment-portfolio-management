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

def filter_by_date(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    return df.loc[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))]

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
