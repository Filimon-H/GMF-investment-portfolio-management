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
