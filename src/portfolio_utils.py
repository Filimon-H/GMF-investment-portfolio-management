# src/portfolio/portfolio_utils.py

import pandas as pd
import numpy as np
from typing import Optional, Dict
def load_asset_data(tsla_path, spy_path, bnd_path):
    """Load processed data for TSLA, SPY, BND."""
    tsla = pd.read_csv(tsla_path, parse_dates=["Date"], index_col="Date")
    spy = pd.read_csv(spy_path, parse_dates=["Date"], index_col="Date")
    bnd = pd.read_csv(bnd_path, parse_dates=["Date"], index_col="Date")
    return tsla, spy, bnd

def compute_daily_returns(df):
    """Compute daily returns from Close prices."""
    return df["Close"].pct_change().dropna()

def calculate_expected_returns(tsla_expected_annual, spy_returns, bnd_returns):
    """Calculate expected annual returns for TSLA, SPY, BND."""
    mu_spy = spy_returns.mean() * 252
    mu_bnd = bnd_returns.mean() * 252
    return {
        "TSLA": tsla_expected_annual,
        "SPY": mu_spy,
        "BND": mu_bnd
    }

def compute_covariance_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the covariance matrix of daily returns."""
    return returns_df.cov()

import numpy as np

def simulate_random_portfolios(expected_returns: dict, cov_matrix: pd.DataFrame, n_portfolios=5000, risk_free_rate=0.01):
    np.random.seed(42)
    tickers = list(expected_returns.keys())
    results = {
        "Returns": [],
        "Volatility": [],
        "Sharpe": [],
        "Weights": []
    }

    for _ in range(n_portfolios):
        weights = np.random.dirichlet(np.ones(len(tickers)), size=1)[0]
        weights = np.array(weights)
        portfolio_return = np.dot(weights, [expected_returns[t] for t in tickers])
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol

        results["Returns"].append(portfolio_return)
        results["Volatility"].append(portfolio_vol)
        results["Sharpe"].append(sharpe_ratio)
        results["Weights"].append(weights)

    return pd.DataFrame(results), tickers


import matplotlib.pyplot as plt

def plot_efficient_frontier(portfolios_df, asset_labels, save_path=None):
    """Plot the Efficient Frontier and highlight key portfolios."""
    plt.figure(figsize=(12, 6))

    # Scatter portfolios by risk (x) and return (y), colored by Sharpe
    scatter = plt.scatter(
        portfolios_df["Volatility"],
        portfolios_df["Returns"],
        c=portfolios_df["Sharpe"],
        cmap="viridis",
        alpha=0.6
    )
    plt.colorbar(scatter, label="Sharpe Ratio")

    # Identify optimal portfolios
    max_sharpe_idx = portfolios_df["Sharpe"].idxmax()
    min_vol_idx = portfolios_df["Volatility"].idxmin()

    # Plot Max Sharpe
    plt.scatter(
        portfolios_df.loc[max_sharpe_idx, "Volatility"],
        portfolios_df.loc[max_sharpe_idx, "Returns"],
        color="red", marker="*", s=200, label="Max Sharpe Ratio"
    )

    # Plot Min Volatility
    plt.scatter(
        portfolios_df.loc[min_vol_idx, "Volatility"],
        portfolios_df.loc[min_vol_idx, "Returns"],
        color="blue", marker="X", s=150, label="Min Volatility"
    )

    plt.title("Efficient Frontier of Random Portfolios")
    plt.xlabel("Annualized Volatility (Risk)")
    plt.ylabel("Expected Annual Return")
    plt.legend()

    

    plt.show()

    return max_sharpe_idx, min_vol_idx




def compute_max_return_weights(mu: pd.Series, caps: Optional[Dict[str, float]] = None) -> np.ndarray:
    """
    Maximize expected return under long-only, sum-to-1.
    Default (no caps): allocate 100% to the asset with the largest μ.
    With caps: greedy fill from highest μ down, respecting caps (0..1).
    """
    tickers = list(mu.index)
    n = len(tickers)
    w = np.zeros(n, dtype=float)

    if not caps:
        w[np.argmax(mu.values)] = 1.0
        return w

    # Greedy with caps
    remaining = 1.0
    # order by descending mu
    order = np.argsort(-mu.values)
    for i in order:
        t = tickers[i]
        cap = float(caps.get(t, 1.0))
        alloc = min(cap, remaining)
        w[i] = alloc
        remaining -= alloc
        if remaining <= 1e-9:
            break
    # if caps too tight, spread remainder uniformly (optional)
    if remaining > 1e-9:
        free_idx = [i for i in range(n) if w[i] < caps.get(tickers[i], 1.0)]
        if free_idx:
            bump = remaining / len(free_idx)
            for i in free_idx:
                room = caps.get(tickers[i], 1.0) - w[i]
                w[i] += min(bump, max(0.0, room))
    # normalize
    s = w.sum()
    return w / s if s > 0 else np.ones(n)/n
