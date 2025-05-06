import os
import pandas as pd
import requests
from core.core_data import fetch_klines, add_indicators

DATA_DIR = 'data/historical'

def get_top_coinex_symbols(limit=250):
    """
    Get top symbols by volume from CoinEx.
    """
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
                volume_usd = vol * last
                pairs_and_volumes.append((symbol, volume_usd))
            except Exception:
                continue

    sorted_pairs = sorted(pairs_and_volumes, key=lambda x: x[1], reverse=True)
    top_symbols = [pair for pair, _ in sorted_pairs[:limit]]
    return top_symbols

def update_historical_data(symbol, interval='5m', limit=1000):
    """
    Update or create historical data CSV for the symbol.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, f"{symbol}_{interval}_{limit}.csv")

    if not os.path.exists(file_path):
        df = fetch_klines(symbol, interval=interval, limit=limit)
        if df.empty:
            raise ValueError(f"No data fetched for {symbol}.")
        df = add_indicators(df)
        df.to_csv(file_path, index=False)
        return df

    # CSV exists — load and update
    existing = pd.read_csv(file_path)

    latest_df = fetch_klines(symbol, interval=interval, limit=limit)
    if latest_df.empty:
        return existing

    # Check if update is needed
    if latest_df["Close"].iloc[-1] != existing["Close"].iloc[-1]:
        latest_df = add_indicators(latest_df)
        latest_df.to_csv(file_path, index=False)
        return latest_df

    # Always ensure ATR exists
    if "ATR" not in existing.columns or existing["ATR"].isnull().all():
        existing = add_indicators(existing)
        existing.to_csv(file_path, index=False)

    return existing

def load_data(symbol, interval="5m", limit=1000):
    """
    Load historical CSV data with indicators.
    """
    file_path = os.path.join(DATA_DIR, f"{symbol}_{interval}_{limit}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file for {symbol} not found. Run update_historical_data first.")
    df = pd.read_csv(file_path)

    # Always check indicators present
    if "ATR" not in df.columns or df["ATR"].isnull().all():
        df = add_indicators(df)
        df.to_csv(file_path, index=False)

    return df
