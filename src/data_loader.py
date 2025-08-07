import yfinance as yf
import pandas as pd
from typing import List, Tuple

def fetch_asset_data(tickers: List[str], start_date: str, end_date: str) -> Tuple[dict, pd.DataFrame]:
    """
    Fetch historical data for a list of asset tickers using yfinance.

    Parameters:
        tickers (List[str]): List of asset symbols to fetch (e.g., ['TSLA', 'SPY', 'BND'])
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format

    Returns:
        Tuple[dict, pd.DataFrame]: 
            - dict of DataFrames for each ticker
            - merged DataFrame with adjusted close prices for all assets
    """
    data = {}
    adj_close = pd.DataFrame()

    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date)
        df['Ticker'] = ticker
        data[ticker] = df
        adj_close[ticker] = df['Adj Close']
    
    adj_close.index.name = 'Date'
    return data, adj_close
