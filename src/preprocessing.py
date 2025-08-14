import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

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


#def normalize_close_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the 'Close' column using Min-Max Scaling.

    Returns a DataFrame with a new 'Close_Normalized' column.
    """
    scaler = MinMaxScaler()
    df = df.copy()
    df["Close_Normalized"] = scaler.fit_transform(df[["Close"]])
    return df






def normalize_close_column(df: pd.DataFrame, save_scaler: bool = False, scaler_filename: str = None) -> pd.DataFrame:
    """
    Normalize the 'Close' column using Min-Max Scaling.
    
    If save_scaler=True and scaler_filename is provided, 
    the fitted scaler will be saved under models/saved/.
    
    Returns a DataFrame with a new 'Close_Normalized' column.
    """
    scaler = MinMaxScaler()
    df = df.copy()
    df["Close_Normalized"] = scaler.fit_transform(df[["Close"]])
    
    if save_scaler and scaler_filename:
        # Build absolute path to models/saved/
        base_dir = os.path.dirname(os.path.dirname(__file__))  # go up from src/ to project root
        save_dir = os.path.join(base_dir, "models", "saved")
        os.makedirs(save_dir, exist_ok=True)  # ensure directory exists
        
        scaler_path = os.path.join(save_dir, scaler_filename)
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")
    
    return df

