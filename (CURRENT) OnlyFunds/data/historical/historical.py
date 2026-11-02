"""Fetch candles from CoinEx into the local cache (data/cache/) for offline backtests."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd  # noqa: E402

from core.core_data import fetch_klines, load_cached  # noqa: E402


def save_historical_data(pair: str, interval: str = "5m", limit: int = 1000) -> pd.DataFrame:
    """Fetch and cache candles; each run appends new bars to the cached history."""
    df = fetch_klines(pair, interval, limit)
    print(f"{pair} {interval}: {len(df)} bars ({df.attrs.get('source', '?')})")
    return df


def load_historical_data(pair: str, interval: str = "5m") -> pd.DataFrame:
    return load_cached(pair, interval)


if __name__ == "__main__":
    save_historical_data("BTCUSDT", "5m", 1000)
