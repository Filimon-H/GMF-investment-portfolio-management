import pandas as pd
import numpy as np

def simulate_portfolio(df_returns: pd.DataFrame, weights: dict, initial_value: float = 1_000) -> pd.Series:
    """
    Simulate portfolio value over time using daily returns and given weights.
    """
    rets = df_returns[list(weights.keys())]
    weighted_returns = rets.dot(np.array(list(weights.values())))
    portfolio_value = (1 + weighted_returns).cumprod() * initial_value
    return portfolio_value


def calculate_sharpe_ratio(portfolio_returns: pd.Series, risk_free_rate: float = 0.01) -> float:
    """
    Calculate Sharpe Ratio for the portfolio.
    """
    excess_daily_returns = portfolio_returns.pct_change() - (risk_free_rate / 252)
    return (excess_daily_returns.mean() / excess_daily_returns.std()) * np.sqrt(252)
