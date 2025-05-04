import logging
import os
import requests
import pandas as pd
import ta
import yaml
from utils.helpers import get_volatile_pairs

# Load YAML config
with open('config/config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

TRADING_PAIRS = CONFIG.get("TRADING_PAIRS", ["BTCUSDT", "ETHUSDT", "LTCUSDT"])
COINEX_BASE = CONFIG.get("API_BASE_URL", "https://api.coinex.com/v1")

_INTERVAL_MAP = {
    "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min",
    "30m": "30min", "1h": "1hour", "2h": "2hour", "4h": "4hour",
    "6h": "6hour", "12h": "12hour", "1d": "1day", "3d": "3day",
    "1w": "1week"
}

def fetch_klines(pair: str, interval: str = "5m", limit: int = 500) -> pd.DataFrame:
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

        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ]).drop(columns=["turnover"])

        df.rename(columns={
            "timestamp": "Timestamp",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume"
        }, inplace=True)

        df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s")
        df.set_index("Timestamp", inplace=True)

        return df[["Open", "High", "Low", "Close", "Volume"]].astype(float)

    except requests.exceptions.RequestException as e:
        logging.error(f"RequestException in fetch_klines for {pair}: {e}")
    except ValueError as e:
        logging.error(f"ValueError in fetch_klines for {pair}, possibly malformed data: {e}")

    return pd.DataFrame()

def validate_df(df: pd.DataFrame) -> bool:
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing_columns = required - set(df.columns)
    if missing_columns:
        logging.error(f"validate_df missing columns: {missing_columns}")
        return False
    if df[list(required)].isnull().any().any():
        logging.error("validate_df found NaNs in OHLCV")
        return False
    return True

def add_indicators(df: pd.DataFrame, indicator_params=None) -> pd.DataFrame:
    df2 = df.copy()
    close = df2["Close"]

    indicator_params = indicator_params or {}
    rsi_window = indicator_params.get("rsi_window", 14)
    macd_fast = indicator_params.get("macd_fast", 12)
    macd_slow = indicator_params.get("macd_slow", 26)
    macd_signal = indicator_params.get("macd_signal", 9)

    df2["rsi"] = ta.momentum.RSIIndicator(close, window=rsi_window).rsi()
    macd = ta.trend.MACD(close, macd_fast, macd_slow, macd_signal)
    df2["macd"] = macd.macd()
    df2["macd_signal"] = macd.macd_signal()
    df2["atr"] = ta.volatility.AverageTrueRange(
        df2["High"], df2["Low"], close, window=14
    ).average_true_range()

    # Always add ATR as "ATR" column for consistency
    df2["ATR"] = df2["atr"]

    return df2
