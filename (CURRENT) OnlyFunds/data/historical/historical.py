import sys
import os

# Ensure the core directory is on the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

import pandas as pd
from datetime import datetime
from core.core_data import fetch_klines, validate_df, add_indicators

HISTORICAL_DATA_DIR = "data/historical"

def save_historical_data(pair: str, interval: str = '5m', limit: int = 1000) -> pd.DataFrame:
    """
    Fetch historical data for the selected trading pair, compute indicators, and save it to a CSV.
    :param pair: Trading pair (e.g., 'BTCUSDT')
    :param interval: Time interval (e.g., '5m')
    :param limit: Number of data points to fetch
    :return: DataFrame containing OHLCV + indicators
    """
    filename = f"{pair}_{interval}_{limit}.csv"
    file_path = os.path.join(HISTORICAL_DATA_DIR, filename)

    # Check if the data file already exists
    if os.path.exists(file_path):
        print(f"Data already cached: {file_path}")
        df = pd.read_csv(file_path)
    else:
        # Fetch new data from the exchange
        print(f"Fetching new data for {pair} at {interval}...")
        df = fetch_klines(pair=pair, interval=interval, limit=limit)

        # Debugging the fetched data
        if df.empty:
            print(f"No data fetched for {pair} at {interval}. Please check the pair and interval.")
            return pd.DataFrame()

        print(f"Fetched data for {pair} at {interval}: {len(df)} rows.")

        # Save the raw OHLCV data
        try:
            df.to_csv(file_path, index=False)
            print(f"Raw OHLCV data saved: {file_path}")
        except Exception as e:
            print(f"Error saving raw OHLCV data for {pair}: {e}")
            return pd.DataFrame()

    # Validate the fetched data
    if not validate_df(df):
        print(f"Data for {pair} failed validation.")
        return pd.DataFrame()

    # Add technical indicators like RSI, MACD, etc.
    df = add_indicators(df)

    # Add Z-score normalized features (for machine learning)
    for col in ["rsi", "macd", "ema_diff", "volatility"]:
        if col in df.columns:
            df[f"{col}_z"] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)

    # Create the supervised target 'indicator' (1 if price is up in the next 5 periods)
    df["indicator"] = 0
    df.loc[df["Close"].shift(-5) > df["Close"], "indicator"] = 1  # Up in 5 periods = positive signal

    final_cols = ["Open", "High", "Low", "Close", "Volume",
                  "rsi", "macd", "ema_diff", "volatility",
                  "rsi_z", "macd_z", "ema_diff_z", "volatility_z",
                  "indicator"]

    df = df.dropna()  # Drop NaN values
    df = df[final_cols]

    # Save the processed data with indicators
    try:
        df.to_csv(file_path, index=False)
        print(f"✅ Processed data saved with indicators: {file_path}")
    except Exception as e:
        print(f"Error saving processed data for {pair}: {e}")
        return pd.DataFrame()

    return df

def load_historical_data(filepath: str = None) -> pd.DataFrame:
    """
    Load a saved historical CSV or all files in the historical folder.
    """
    if filepath:
        if os.path.exists(filepath):
            print(f"Loading file: {filepath}")
            return pd.read_csv(filepath)
        else:
            print(f"File not found: {filepath}")
            return pd.DataFrame()

    # Load all historical CSVs in the directory
    all_files = [f for f in os.listdir(HISTORICAL_DATA_DIR) if f.endswith(".csv")]
    dfs = []
    for f in all_files:
        file_path = os.path.join(HISTORICAL_DATA_DIR, f)
        try:
            print(f"Loading file: {file_path}")
            df = pd.read_csv(file_path)
            dfs.append(df)
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")

    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        print(f"✅ Loaded {len(combined)} rows from {len(dfs)} files.")
        return combined
    else:
        print("❗ No historical data files found.")
        return pd.DataFrame()

# Example usage: dynamically fetch and save data for the pair, interval, and limit
if __name__ == "__main__":
    pair = "BTCUSDT"  # This can be dynamically set based on the app's selected pair
    interval = "5m"  # This can be dynamically set based on the app's selected interval
    limit = 1000      # Number of data points to fetch
    save_historical_data(pair, interval, limit)
