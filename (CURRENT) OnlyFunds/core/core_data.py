"""Market data access: CoinEx candles with a local CSV cache.

``fetch_klines`` pulls completed candles from CoinEx. When the network is
unavailable it falls back to the most recent cached copy, and says so via
``df.attrs["source"]`` so the UI can show where data came from.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

import pandas as pd

from core.coinex_client import CoinExClient, CoinExError, INTERVAL_MAP
from core.indicators import add_indicators, validate_ohlcv  # re-exported

logger = logging.getLogger(__name__)

APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE_DIR = os.path.join(APP_ROOT, "data", "cache")
DEFAULT_PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "LTCUSDT", "BNBUSDT"]

_client: Optional[CoinExClient] = None


def get_client() -> CoinExClient:
    global _client
    if _client is None:
        _client = CoinExClient()
    return _client


def get_trading_pairs() -> List[str]:
    try:
        from utils.config import load_config
        pairs = load_config().trading_pairs
        if pairs and isinstance(pairs, list):
            return pairs
    except Exception:
        pass
    return list(DEFAULT_PAIRS)


TRADING_PAIRS = get_trading_pairs()


def _cache_path(pair: str, interval: str) -> str:
    return os.path.join(CACHE_DIR, f"{pair}_{interval}.csv")


def save_data(df: pd.DataFrame, pair: str, interval: str = "5m") -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(pair, interval)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def load_cached(pair: str, interval: str = "5m", limit: Optional[int] = None) -> pd.DataFrame:
    path = _cache_path(pair, interval)
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if limit:
        df = df.tail(int(limit)).reset_index(drop=True)
    df.attrs["source"] = "cache"
    return df


def fetch_klines(pair: str, interval: str = "5m", limit: int = 1000, use_cache_fallback: bool = True,
                 client: Optional[CoinExClient] = None) -> pd.DataFrame:
    """Completed OHLCV candles for ``pair`` (oldest first)."""
    if interval not in INTERVAL_MAP:
        raise ValueError(f"Unsupported interval {interval}")
    client = client or get_client()
    try:
        df = client.get_klines(pair, interval=interval, limit=limit)
        if df.empty:
            raise CoinExError(None, f"no candles returned for {pair}")
        merged = _merge_with_cache(df, pair, interval)
        save_data(merged, pair, interval)
        out = merged.tail(int(limit)).reset_index(drop=True)
        out.attrs["source"] = "coinex"
        return out
    except Exception as exc:
        if not use_cache_fallback:
            raise
        logger.warning("Live kline fetch failed for %s %s (%s); using cache", pair, interval, exc)
        cached = load_cached(pair, interval, limit)
        if cached.empty:
            raise
        return cached


def _merge_with_cache(df: pd.DataFrame, pair: str, interval: str) -> pd.DataFrame:
    """Append new candles to the cached history so longer backtests are possible."""
    cached = load_cached(pair, interval)
    if cached.empty or "created_at" not in cached.columns:
        return df
    merged = pd.concat([cached, df], ignore_index=True)
    merged = merged.drop_duplicates("created_at", keep="last").sort_values("created_at")
    return merged.reset_index(drop=True)


def load_data(pair: str, interval: str = "5m", limit: int = 1000) -> pd.DataFrame:
    return fetch_klines(pair, interval, limit)


def validate_df(df: Optional[pd.DataFrame]) -> bool:
    return validate_ohlcv(df)


__all__ = ["fetch_klines", "load_data", "load_cached", "save_data", "validate_df", "add_indicators",
           "TRADING_PAIRS", "get_client", "DEFAULT_PAIRS"]
