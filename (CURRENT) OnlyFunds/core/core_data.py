import os
import pandas as pd

def get_trading_pairs():
    """
    Loads trading pairs from config if available, else returns a default list.
    """
    try:
        from utils.config import Config
        config_pairs = Config().trading_pairs
        if config_pairs and isinstance(config_pairs, list):
            return config_pairs
    except Exception:
        pass
    # Fallback default pairs
    return ["BTCUSDT", "ETHUSDT", "LTCUSDT", "SOLUSDT", "BNBUSDT"]

TRADING_PAIRS = get_trading_pairs()

def fetch_klines(pair, interval="5m", limit=1000):
    """
    Fetch klines for a trading pair from local CSV file.
    Fallback is an empty DataFrame if file not found.
    """
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    fname = f"{pair}_{interval}_{limit}.csv"
    fpath = os.path.join(data_dir, fname)
    if os.path.exists(fpath):
        df = pd.read_csv(fpath)
    else:
        df = pd.DataFrame()
    return df

def load_data(pair, interval="5m", limit=1000):
    """
    Loads historical data for a trading pair from a CSV file.
    """
    return fetch_klines(pair, interval, limit)

def save_data(df, pair, interval="5m", limit=1000):
    """
    Saves a DataFrame to a CSV in the data directory.
    """
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    fname = f"{pair}_{interval}_{limit}.csv"
    fpath = os.path.join(data_dir, fname)
    df.to_csv(fpath, index=False)

def validate_df(df):
    """
    Validates that a DataFrame is non-empty and contains 'Close' and 'ATR' columns.
    """
    if df is None or df.empty:
        return False
    for col in ["Close", "ATR"]:
        if col not in df.columns:
            return False
    return True

def add_indicators(df):
    """
    Adds 'ATR' and ensures 'Close' column exists in DataFrame.
    Uses a rolling std as a fallback for ATR.
    """
    if df is None or df.empty:
        return df

    # Add Close if missing (dummy values for robustness)
    if "Close" not in df.columns:
        df["Close"] = 100 + pd.Series(range(len(df)))

    # ATR: Average True Range (basic rolling window as fallback)
    if "ATR" not in df.columns:
        df["ATR"] = df["Close"].rolling(window=14, min_periods=1).std().fillna(0.0)

    return df

__all__ = [
    "fetch_klines",
    "load_data",
    "save_data",
    "validate_df",
    "add_indicators",
    "TRADING_PAIRS",
]