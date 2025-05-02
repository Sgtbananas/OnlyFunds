# historical.py

import os
import pandas as pd
from datetime import datetime
from core.core_data import fetch_klines, validate_df, add_indicators

HISTORICAL_DATA_DIR = "data/historical"

def save_historical_data(pair: str, interval: str, limit: int = 1000) -> pd.DataFrame:
    """
    Fetch historical data, compute indicators, and save it to a CSV.
    :param pair: Trading pair (e.g., 'BTCUSDT')
    :param interval: Time interval (e.g., '5m')
    :param limit: Number of data points to fetch
    :return: DataFrame containing OHLCV + indicators
    """
    filename = f"{pair}_{interval}_{limit}.csv"
    file_path = os.path.join(HISTORICAL_DATA_DIR, filename)

    if os.path.exists(file_path):
        print(f"Data already cached: {file_path}")
        df = pd.read_csv(file_path)
    else:
        # Fetch new data
        df = fetch_klines(pair=pair, interval=interval, limit=limit)
        if df.empty:
            print(f"No data fetched for {pair} at {interval}.")
            return pd.DataFrame()

        df.to_csv(file_path)
        print(f"Raw OHLCV data saved: {file_path}")

    # Validate and add indicators
    if not validate_df(df):
        print(f"Data for {pair} failed validation.")
        return pd.DataFrame()

    df = add_indicators(df)

    # Add Z-score normalized features
    for col in ["rsi", "macd", "ema_diff", "volatility"]:
        if col in df.columns:
            df[f"{col}_z"] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)

    # Create the supervised target 'indicator'
    df["indicator"] = 0
    df.loc[df["Close"].shift(-5) > df["Close"], "indicator"] = 1  # Up in 5 periods = positive signal

    final_cols = ["Open", "High", "Low", "Close", "Volume",
                  "rsi", "macd", "ema_diff", "volatility",
                  "rsi_z", "macd_z", "ema_diff_z", "volatility_z",
                  "indicator"]

    df = df.dropna()
    df = df[final_cols]
    df.to_csv(file_path, index=False)
    print(f"✅ Processed data saved with indicators: {file_path}")

    return df

def load_historical_data(filepath: str = None) -> pd.DataFrame:
    """
    Load a saved historical CSV or all files in the historical folder.
    """
    if filepath:
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        else:
            print(f"File not found: {filepath}")
            return pd.DataFrame()

    # Load all historical CSVs
    all_files = [f for f in os.listdir(HISTORICAL_DATA_DIR) if f.endswith(".csv")]
    dfs = []
    for f in all_files:
        try:
            df = pd.read_csv(os.path.join(HISTORICAL_DATA_DIR, f))
            dfs.append(df)
        except Exception as e:
            print(f"Failed to load {f}: {e}")

    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        print(f"✅ Loaded {len(combined)} rows from {len(dfs)} files.")
        return combined
    else:
        print("❗ No historical data files found.")
        return pd.DataFrame()
