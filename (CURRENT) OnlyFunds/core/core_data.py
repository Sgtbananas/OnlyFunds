import pandas as pd
import numpy as np
import os
import logging
import time
import requests
from datetime import datetime
from utils.helpers import add_indicators

logger = logging.getLogger(__name__)

DATA_DIR = "data/historical"
os.makedirs(DATA_DIR, exist_ok=True)

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

    # Always add indicators on fetch
    df = add_indicators(df)

    return df
def save_data(df, pair, interval="5m", limit=1000):
    """
    Save DataFrame to CSV in the historical data directory.
    """
    filename = f"{pair}_{interval}_{limit}.csv"
    filepath = os.path.join(DATA_DIR, filename)
    df.to_csv(filepath)
    logger.info(f"Saved data for {pair} to {filepath}.")


def load_data(pair, interval="5m", limit=1000):
    """
    Load historical data from CSV. If missing, fetches new data.
    Always ensures indicators are added.
    """
    filename = f"{pair}_{interval}_{limit}.csv"
    filepath = os.path.join(DATA_DIR, filename)

    if os.path.exists(filepath):
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath, index_col="Timestamp", parse_dates=True)
        # Ensure indicators exist
        df = add_indicators(df)
        return df
    else:
        logger.warning(f"Data file not found for {pair}, fetching new data.")
        df = fetch_klines(pair, interval, limit)
        if not df.empty:
            save_data(df, pair, interval, limit)
            return df
        else:
            logger.error(f"Failed to fetch data for {pair}. Returning empty DataFrame.")
            return pd.DataFrame()
