# src/portfolio/portfolio_utils.py

import pandas as pd

def load_asset_data(tsla_path, spy_path, bnd_path):
    """Load processed data for TSLA, SPY, BND."""
    tsla = pd.read_csv(tsla_path, parse_dates=["Date"], index_col="Date")
    spy = pd.read_csv(spy_path, parse_dates=["Date"], index_col="Date")
    bnd = pd.read_csv(bnd_path, parse_dates=["Date"], index_col="Date")
    return tsla, spy, bnd

def compute_daily_returns(df):
    """Compute daily returns from Close prices."""
    return df["Close"].pct_change().dropna()
