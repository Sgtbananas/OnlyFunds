"""Deterministic synthetic OHLCV data for tests (seeded, no network)."""
import numpy as np
import pandas as pd


def synthetic_ohlcv(n: int = 3000, seed: int = 1, drift: float = 0.0, vol: float = 0.004,
                    cycle: float = 0.0004, freq: str = "5min", start: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, vol, n) + cycle * np.sin(np.arange(n) / 150)
    close = start * np.exp(np.cumsum(r))
    open_ = np.r_[start, close[:-1]]
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, n)))
    ts = pd.date_range("2025-01-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"timestamp": ts, "created_at": ts.asi8 // 10**6, "Open": open_, "High": high,
                         "Low": low, "Close": close, "Volume": rng.uniform(1, 10, n)})
