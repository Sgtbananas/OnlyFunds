# historical.py

import os
import pandas as pd
from datetime import datetime
from core_data import fetch_klines

HISTORICAL_DATA_DIR = "data/historical"

def save_historical_data(pair: str, interval: str, limit: int = 1000) -> pd.DataFrame:
    """
    Fetch historical data and save it to a CSV file.
    :param pair: Trading pair (e.g., 'BTCUSDT')
    :param interval: Time interval (e.g., '5m')
    :param limit: Number of data points to fetch
    :return: DataFrame containing OHLCV data
    """
    filename = f"{pair}_{interval}_{limit}.csv"
    file_path = os.path.join(HISTORICAL_DATA_DIR, filename)

    if os.path.exists(file_path):
        print(f"Data already cached: {file_path}")
        return pd.read_csv(file_path)

    # Fetch new data
    df = fetch_klines(pair=pair, interval=interval, limit=limit)
    if not df.empty:
        df.to_csv(file_path)
        print(f"Data saved: {file_path}")
    return df
