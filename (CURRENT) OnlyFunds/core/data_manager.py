
import os
import pandas as pd
import requests
from core.core_data import fetch_klines

DATA_DIR = 'data/historical'

def get_top_coinex_symbols(limit=250):
    url = "https://api.coinex.com/v1/market/list"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch CoinEx markets: {response.text}")

    all_pairs = response.json().get("data", [])
    spot_pairs = []
    for pair in all_pairs:
        if ":" not in pair and any(q in pair for q in ["USDT", "BTC", "ETH"]):
            spot_pairs.append(pair)

    spot_pairs = sorted(spot_pairs)[:limit]
    return spot_pairs

def update_historical_data(symbol, interval='5m', limit=1000):
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, f"{symbol}_{interval}_{limit}.csv")

    # If no CSV exists, fetch fresh data
    if not os.path.exists(file_path):
        df = fetch_klines(symbol, interval=interval, limit=limit)
        if df.empty:
            raise ValueError(f"No data fetched for {symbol}.")
        df.to_csv(file_path, index=False)
        return df

    # CSV exists — load it
    existing = pd.read_csv(file_path)

    # Fetch latest data
    latest_df = fetch_klines(symbol, interval=interval, limit=limit)
    if latest_df.empty:
        return existing  # Keep using the old data

    # If the new data has timestamps newer than the existing data, update
    latest_close = latest_df["Close"].iloc[-1]
    existing_close = existing["Close"].iloc[-1]

    if latest_close != existing_close:
        latest_df.to_csv(file_path, index=False)
        return latest_df
    else:
        return existing  # Data is already up to date

def load_data(symbol, interval="5m", limit=1000):
    file_path = os.path.join(DATA_DIR, f"{symbol}_{interval}_{limit}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file for {symbol} not found. Run update_historical_data first.")
    return pd.read_csv(file_path)
