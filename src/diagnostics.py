from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
sys.path.append(os.path.abspath('../src'))
from utils import save_plot  # ✅ Import save utility

def run_adf_test(series: pd.Series, asset_name: str) -> None:
    print(f"\n🔍 ADF Test for {asset_name}")
    result = adfuller(series.dropna())
    print(f"Test Statistic: {result[0]:.4f}")
    print(f"P-value: {result[1]:.4f}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"  {key}: {value:.4f}")
    if result[1] < 0.05:
        print("✅ Likely Stationary (p < 0.05)")
    else:
        print("⚠️ Likely Non-Stationary (p ≥ 0.05)")


def calculate_var(series: pd.Series, confidence: float = 0.95) -> float:
    return -np.percentile(series.dropna(), (1 - confidence) * 100)


def calculate_sharpe_ratio(series: pd.Series, risk_free_rate: float = 0.01) -> float:
    daily_excess_return = series - (risk_free_rate / 252)
    mean_return = daily_excess_return.mean()
    std_return = daily_excess_return.std()
    return (mean_return / std_return) * np.sqrt(252)


def plot_return_distribution_with_var(series: pd.Series, asset_name: str, confidence: float = 0.95) -> None:
    """
    Plot histogram of daily returns and show Value at Risk (VaR) as a vertical line.
    Also saves the figure to results/plots.
    """
    var_value = calculate_var(series, confidence=confidence)
    
    plt.figure(figsize=(10, 5))
    sns.histplot(series.dropna(), bins=100, kde=True, color='skyblue')
    plt.axvline(-var_value, color='red', linestyle='--', label=f'VaR {int(confidence*100)}% = {-var_value:.4f}')
    plt.title(f"{asset_name} Daily Return Distribution with {int(confidence*100)}% VaR")
    plt.xlabel("Daily Return")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    
    # ✅ Save the plot
    filename = f"{asset_name.lower()}_return_distribution.png"
    save_plot(filename)
    
    plt.show()
