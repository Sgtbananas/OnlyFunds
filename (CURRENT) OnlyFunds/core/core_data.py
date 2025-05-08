import pandas as pd
import numpy as np
import os
import logging
import requests
from datetime import datetime

logger = logging.getLogger(__name__)

DATA_DIR = "data/historical"
os.makedirs(DATA_DIR, exist_ok=True)

def add_indicators(df):
    """
    Adds ATR and any other indicators to the dataframe.
    """
    import ta

    if "ATR" not in df.columns:
        try:
            high = df["High"]
            low = df["Low"]
            close = df["Close"]
            df["ATR"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
        except Exception as e:
            logger.warning(f"[add_indicators] Failed to compute ATR: {e}")
            df["ATR"] = np.nan

    return df

def validate_df(df):
    return df is not None and not df.empty and "Close" in df.columns

def fetch_klines(pair, interval="5m", limit=1000):
    """
    Fetch historical kline/candle data from CoinEx.
    """
    url = f"https://api.coinex.com/v1/market/kline?market={pair}&type={interval}&limit={limit}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()["data"]
    except Exception as e:
        logger.error(f"Failed to fetch klines for {pair}: {e}")
        return pd.DataFrame()

    if not data:
        logger.warning(f"No kline data returned for {pair}.")
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s")
    df.set_index("Timestamp", inplace=True)
    df = df.astype(float)

    df = add_indicators(df)

    return df

def save_data(df, pair, interval="5m", limit=1000):
    """
    Save DataFrame to CSV.
    """
    filename = f"{pair}_{interval}_{limit}.csv"
    filepath = os.path.join(DATA_DIR, filename)
    df.to_csv(filepath)
    logger.info(f"Saved data for {pair} to {filepath}.")

def load_data(pair, interval="5m", limit=1000):
    """
    Load CSV. If missing or missing ATR, fetches fresh data.
    """
    filename = f"{pair}_{interval}_{limit}.csv"
    filepath = os.path.join(DATA_DIR, filename)

    if os.path.exists(filepath):
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath, index_col="Timestamp", parse_dates=True)
        df = add_indicators(df)
        return df
    else:
        logger.warning(f"No data file found for {pair}. Fetching new data.")
        df = fetch_klines(pair, interval, limit)
        if not df.empty:
            save_data(df, pair, interval, limit)
            return df
        else:
            logger.error(f"Failed to fetch data for {pair}.")
            return pd.DataFrame()
