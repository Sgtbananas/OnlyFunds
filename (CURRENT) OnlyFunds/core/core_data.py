# core/core_data.py

import logging
import os
import requests
import pandas as pd
import ta
from dotenv import load_dotenv

load_dotenv()

# Interval mapping for CoinEx API
_INTERVAL_MAP = {
    "1m":  "1min",  "3m":  "3min", "5m":  "5min", "15m": "15min",
    "30m": "30min", "1h":  "1hour","2h":  "2hour","4h":  "4hour",
    "6h":  "6hour","12h": "12hour","1d":  "1day","3d":  "3day",
    "1w":  "1week"
}

# Default trading pairs
TRADING_PAIRS = os.getenv("TRADING_PAIRS", "BTCUSDT,ETHUSDT,LTCUSDT").split(",")

COINEX_BASE = os.getenv("API_BASE_URL", "https://api.coinex.com/v1")

def fetch_klines(pair: str, interval: str = "5m", limit: int = 500) -> pd.DataFrame:
    """
    Fetch OHLCV data from CoinEx. Returns DataFrame indexed by Timestamp
    with Open, High, Low, Close, Volume columns.
    """
    resolution = _INTERVAL_MAP.get(interval)
    if not resolution:
        logging.error(f"Invalid interval: {interval}")
        return pd.DataFrame()

    try:
        resp = requests.get(
            f"{COINEX_BASE}/market/kline",
            params={"market": pair, "type": resolution, "limit": limit},
            timeout=15
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            logging.warning(f"No kline data for {pair}@{interval}")
            return pd.DataFrame()

        # CoinEx: [timestamp, open, high, low, close, volume, turnover]
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ]).drop(columns=["turnover"])
        df.rename(columns={
            "timestamp": "Timestamp",
            "open":      "Open",
            "high":      "High",
            "low":       "Low",
            "close":     "Close",
            "volume":    "Volume"
        }, inplace=True)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s")
        df.set_index("Timestamp", inplace=True)
        return df[["Open","High","Low","Close","Volume"]].astype(float)
    except Exception as e:
        logging.error(f"fetch_klines error for {pair}: {e}")
        return pd.DataFrame()

def validate_df(df: pd.DataFrame) -> bool:
    """Ensure DataFrame has OHLCV & no NaNs."""
    required = {"Open","High","Low","Close","Volume"}
    if not required.issubset(df.columns):
        logging.error(f"validate_df missing columns: {required - set(df.columns)}")
        return False
    if df[list(required)].isnull().any().any():
        logging.error("validate_df found NaNs in OHLCV")
        return False
    return True

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append RSI, MACD, Bollinger Bands, EMA20, EMA_diff, Volatility.
    Fully forward‐fills to avoid NaNs.
    """
    df2 = df.copy()
    close = df2["Close"]

    # RSI
    df2["rsi"] = (
        ta.momentum.RSIIndicator(close, window=14)
        .rsi()
        .fillna(50)
        .ffill()
    )

    # MACD
    macd = ta.trend.MACD(close)
    df2["macd"]        = macd.macd_diff().fillna(0).ffill()
    df2["macd_signal"] = macd.macd_signal().fillna(0).ffill()

    # Bollinger Bands
    mid = close.rolling(20, min_periods=1).mean()
    std = close.rolling(20, min_periods=1).std().fillna(0)
    df2["bollinger_mid"]   = mid.ffill()
    df2["bollinger_upper"] = (mid + 2 * std).ffill()
    df2["bollinger_lower"] = (mid - 2 * std).ffill()

    # EMA 20
    ema20 = close.ewm(span=20, adjust=False).mean()
    df2["ema20"]    = ema20.ffill()
    df2["ema_diff"] = (close - ema20).fillna(0)

    # Volatility
    df2["volatility"] = close.rolling(10, min_periods=1).std().fillna(0)

    return df2
