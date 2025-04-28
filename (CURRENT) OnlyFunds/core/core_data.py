import logging
import os
import requests
import pandas as pd
import ta
from dotenv import load_dotenv

load_dotenv()

_INTERVAL_MAP = {
    "1m":  "1min",  "3m":  "3min", "5m":  "5min", "15m": "15min",
    "30m": "30min", "1h":  "1hour","2h":  "2hour","4h":  "4hour",
    "6h":  "6hour","12h": "12hour","1d":  "1day","3d":  "3day",
    "1w":  "1week"
}

TRADING_PAIRS = os.getenv("TRADING_PAIRS", "BTCUSDT,ETHUSDT,LTCUSDT").split(",")

COINEX_BASE = os.getenv("API_BASE_URL", "https://api.coinex.com/v1")

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
            "open":      "Open",
            "high":      "High",
            "low":       "Low",
            "close":     "Close",
            "volume":    "Volume"
        }, inplace=True)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s")
        df.set_index("Timestamp", inplace=True)
        return df[["Open","High","Low","Close","Volume"]].astype(float)
    except requests.exceptions.RequestException as e:
        logging.error(f"RequestException in fetch_klines for {pair}: {e}")
    except ValueError as e:
        logging.error(f"ValueError in fetch_klines for {pair}, possibly malformed data: {e}")
    return pd.DataFrame()

def validate_df(df: pd.DataFrame) -> bool:
    required = {"Open","High","Low","Close","Volume"}
    missing_columns = required - set(df.columns)
    if missing_columns:
        logging.error(f"validate_df missing columns: {missing_columns}")
        return False
    if df[list(required)].isnull().any().any():
        logging.error("validate_df found NaNs in OHLCV")
        return False
    return True

def add_indicators(df: pd.DataFrame, indicator_params=None) -> pd.DataFrame:
    """
    Adds indicators to df, optionally using custom params:
    indicator_params: {
        "rsi_window": int,
        "macd_fast": int,
        "macd_slow": int,
        "macd_signal": int,
        "boll_window": int,
        "boll_std": float,
        "ema_span": int,
        "volatility_window": int
    }
    """
    df2 = df.copy()
    close = df2["Close"]
    indicator_params = indicator_params or {}
    rsi_window = indicator_params.get("rsi_window", 14)
    macd_fast = indicator_params.get("macd_fast", 12)
    macd_slow = indicator_params.get("macd_slow", 26)
    macd_signal = indicator_params.get("macd_signal", 9)
    boll_window = indicator_params.get("boll_window", 20)
    boll_std = indicator_params.get("boll_std", 2)
    ema_span = indicator_params.get("ema_span", 20)
    volatility_window = indicator_params.get("volatility_window", 10)

    try:
        df2["rsi"] = (
            ta.momentum.RSIIndicator(close, window=rsi_window)
            .rsi()
            .fillna(50)
            .ffill()
        )
        macd = ta.trend.MACD(
            close, window_slow=macd_slow, window_fast=macd_fast, window_sign=macd_signal
        )
        df2["macd"]        = macd.macd_diff().fillna(0).ffill()
        df2["macd_signal"] = macd.macd_signal().fillna(0).ffill()

        mid = close.rolling(boll_window, min_periods=1).mean()
        std = close.rolling(boll_window, min_periods=1).std().fillna(0)
        df2["bollinger_mid"]   = mid.ffill()
        df2["bollinger_upper"] = (mid + boll_std * std).ffill()
        df2["bollinger_lower"] = (mid - boll_std * std).ffill()

        ema = close.ewm(span=ema_span, adjust=False).mean()
        df2["ema"]      = ema.ffill()
        df2["ema_diff"] = (close - ema).fillna(0)

        df2["volatility"] = close.rolling(volatility_window, min_periods=1).std().fillna(0)

        # Composite z-score indicators for robust ML features
        features = ["rsi", "macd", "ema_diff", "volatility"]
        for col in features:
            mean = df2[col].mean()
            std = df2[col].std(ddof=0)
            if std == 0:
                df2[f"{col}_z"] = 0
            else:
                df2[f"{col}_z"] = (df2[col] - mean) / std
        z_cols = [f"{c}_z" for c in features]
        df2["indicator"] = df2[z_cols].mean(axis=1)
    except Exception as e:
        logging.error(f"Error in add_indicators: {e}")
        return pd.DataFrame()

    return df2
