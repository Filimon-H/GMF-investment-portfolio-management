import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def inspect_asset(df: pd.DataFrame, name: str) -> None:
    """Print basic stats, missing values, data types, and duplicates."""
    print(f"\n📊 {name} Summary")
    print("Shape:", df.shape)
    print("Missing values:\n", df.isnull().sum())
    print("Data types:\n", df.dtypes)
    print("Duplicated rows:", df.duplicated().sum())
    print("Statistical Summary:")
    display(df.describe())


def clean_missing_values(df: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
    """
    Handle missing values with a given method: 'interpolate', 'drop', or 'ffill'.

    Returns cleaned DataFrame.
    """
    if method == 'interpolate':
        return df.interpolate(method='linear')
    elif method == 'drop':
        return df.dropna()
    elif method == 'ffill':
        return df.fillna(method='ffill')
    else:
        raise ValueError("Method must be 'interpolate', 'drop', or 'ffill'")


def normalize_close_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the 'Close' column using Min-Max Scaling.

    Returns a DataFrame with a new 'Close_Normalized' column.
    """
    scaler = MinMaxScaler()
    df = df.copy()
    df["Close_Normalized"] = scaler.fit_transform(df[["Close"]])
    return df
