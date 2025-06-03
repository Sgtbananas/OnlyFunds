import os
import pandas as pd

def get_trading_pairs():
    try:
        from utils.config import Config
        config_pairs = Config().trading_pairs
        if config_pairs and isinstance(config_pairs, list):
            return config_pairs
    except Exception:
        pass
    return ["BTCUSDT", "ETHUSDT", "LTCUSDT", "SOLUSDT", "BNBUSDT"]

TRADING_PAIRS = get_trading_pairs()

def fetch_klines(pair, interval="5m", limit=1000):
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    fname = f"{pair}_{interval}_{limit}.csv"
    fpath = os.path.join(data_dir, fname)
    if os.path.exists(fpath):
        df = pd.read_csv(fpath)
    else:
        df = pd.DataFrame()
    return df

def load_data(pair, interval="5m", limit=1000):
    """Loads historical data for a trading pair from a CSV (stub)."""
    return fetch_klines(pair, interval, limit)

def save_data(df, pair, interval="5m", limit=1000):
    """Saves a DataFrame to a CSV in the data directory (stub)."""
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    fname = f"{pair}_{interval}_{limit}.csv"
    fpath = os.path.join(data_dir, fname)
    df.to_csv(fpath, index=False)

def validate_df(df):
    if df is None or df.empty:
        return False
    for col in ["Close", "ATR"]:
        if col not in df.columns:
            return False
    return True

def add_indicators(df):
    if df is None or df.empty:
        return df
    if "Close" not in df.columns:
        df["Close"] = 100 + pd.Series(range(len(df)))
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