import yfinance as yf
import pandas as pd
from typing import List, Tuple

def fetch_asset_data(tickers: List[str], start_date: str, end_date: str) -> Tuple[dict, pd.DataFrame]:
    """
    Fetch historical data for a list of asset tickers using yfinance.
    Flattens column names and stores 'Close' prices in a single merged dataframe.

    Returns:
        - data: dict of raw DataFrames per ticker
        - close_prices: merged Close price dataframe
    """
    data = {}
    close_prices = pd.DataFrame()

    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if df.empty:
            print(f"⚠️ Warning: No data for {ticker}. Skipping.")
            continue

        # Flatten column names if it's a MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        # Ensure expected columns exist
        if 'Close' not in df.columns:
            print(f"⚠️ 'Close' column missing for {ticker}. Skipping.")
            continue

        # Add Ticker and format index
        df.reset_index(inplace=True)
        df['Ticker'] = ticker
        df.set_index('Date', inplace=True)

        # Save to dictionary
        data[ticker] = df

        # Store Close price
        close_prices[ticker] = df['Close']

    close_prices.index.name = 'Date'
    return data, close_prices
