import os
import pandas as pd

# --- CONFIG LOADING ---
def get_trading_pairs():
    """
    Tries to load trading pairs from config; falls back to default set if unavailable.
    """
    try:
        # Try loading from config if available
        from utils.config import Config
        config_pairs = Config().trading_pairs
        if config_pairs and isinstance(config_pairs, list):
            return config_pairs
    except Exception:
        pass
    # Fallback (safe default pairs)
    return ["BTCUSDT", "ETHUSDT", "LTCUSDT", "SOLUSDT", "BNBUSDT"]

TRADING_PAIRS = get_trading_pairs()

# --- DATA LOADING ---
def fetch_klines(pair, interval="5m", limit=1000):
    """
    Fetch klines for a trading pair. This is a stub; replace this with actual exchange logic.
    """
    # You may want to plug in your real data loader here.
    # For now, we return an empty DataFrame if data not found.
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    fname = f"{pair}_{interval}_{limit}.csv"
    fpath = os.path.join(data_dir, fname)
    if os.path.exists(fpath):
        df = pd.read_csv(fpath)
    else:
        df = pd.DataFrame()
    return df

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
    Dummy indicator adder for robustness; replace with your actual logic.
    Adds 'ATR' and other columns if missing.
    """
    if df is None or df.empty:
        return df

    # Add Close if missing (random values, fallback only for robustness)
    if "Close" not in df.columns:
        df["Close"] = 100 + pd.Series(range(len(df)))

    # ATR: Average True Range (basic rolling window as fallback)
    if "ATR" not in df.columns:
        # ATR requires at least 2 columns, use a rolling std as minimal fallback
        df["ATR"] = df["Close"].rolling(window=14, min_periods=1).std().fillna(0.0)

    return df

# --- EXPORTS ---
__all__ = [
    "fetch_klines",
    "validate_df",
    "add_indicators",
    "TRADING_PAIRS",
]