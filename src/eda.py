import pandas as pd
import matplotlib.pyplot as plt

def plot_close_prices(tsla: pd.DataFrame, spy: pd.DataFrame, bnd: pd.DataFrame) -> None:
    """Plot closing prices for TSLA, SPY, BND."""
    plt.figure(figsize=(14, 6))
    tsla['Close'].plot(label='TSLA')
    spy['Close'].plot(label='SPY')
    bnd['Close'].plot(label='BND')
    plt.title("Closing Prices Over Time")
    plt.ylabel("Price (USD)")
    plt.xlabel("Date")
    plt.legend()
    plt.show()


def compute_returns_and_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily returns, rolling mean, and rolling volatility (30-day)."""
    df = df.copy()
    df['Daily_Return'] = df['Close'].pct_change()
    df['Rolling_Volatility'] = df['Daily_Return'].rolling(window=30).std()
    df['Rolling_Mean'] = df['Daily_Return'].rolling(window=30).mean()
    return df


def plot_volatility_and_mean(df: pd.DataFrame, label: str) -> None:
    """Plot rolling volatility and rolling mean for a single asset."""
    plt.figure(figsize=(14, 6))
    df['Rolling_Volatility'].plot(label=f'{label} Volatility')
    df['Rolling_Mean'].plot(label=f'{label} Rolling Mean')
    plt.title(f"{label}: 30-Day Rolling Mean and Volatility")
    plt.ylabel("Value")
    plt.xlabel("Date")
    plt.legend()
    plt.show()


def detect_return_outliers(df: pd.DataFrame, z_thresh: float = 3.0) -> pd.DataFrame:
    """Detect days with extreme returns based on Z-score."""
    df = df.copy()
    mean = df['Daily_Return'].mean()
    std = df['Daily_Return'].std()
    df['Return_Z'] = (df['Daily_Return'] - mean) / std
    return df[df['Return_Z'].abs() > z_thresh]


def plot_outliers(df: pd.DataFrame, outliers: pd.DataFrame, label: str) -> None:
    """Plot daily returns and highlight outliers."""
    plt.figure(figsize=(14, 5))
    df['Daily_Return'].plot(label=f'{label} Return')
    plt.scatter(outliers.index, outliers['Daily_Return'], color='red', label='Outliers')
    plt.title(f"{label} Daily Returns with Outliers Highlighted")
    plt.ylabel("Daily Return")
    plt.xlabel("Date")
    plt.legend()
    plt.show()
