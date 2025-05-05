
import os
import pandas as pd
import requests
from core.core_data import fetch_klines

DATA_DIR = 'data/historical'

def get_top_coinex_symbols(limit=250):
    # Get all market tickers with volume data
    ticker_url = "https://api.coinex.com/v1/market/ticker/all"
    response = requests.get(ticker_url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch CoinEx ticker data: {response.text}")

    ticker_data = response.json().get("data", {}).get("ticker", {})

    pairs_and_volumes = []

    for symbol, stats in ticker_data.items():
        if ":" not in symbol and any(q in symbol for q in ["USDT", "BTC", "ETH"]):
            try:
                vol = float(stats.get("vol", 0))
                last = float(stats.get("last", 0))
                volume_usd = vol * last  # Approximate 24h trading volume in USD
                pairs_and_volumes.append((symbol, volume_usd))
            except:
                continue

    # Sort descending by volume
    sorted_pairs = sorted(pairs_and_volumes, key=lambda x: x[1], reverse=True)
    top_symbols = [pair for pair, vol in sorted_pairs[:limit]]
    return top_symbols

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
