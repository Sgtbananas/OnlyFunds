"""Candle data quality checks (missing candles, duplicates)."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from core.coinex_client import INTERVAL_SECONDS


def _open_times_ms(df: pd.DataFrame) -> Optional[np.ndarray]:
    if "created_at" in df.columns:
        return df["created_at"].to_numpy("int64")
    if "timestamp" in df.columns:
        return pd.to_datetime(df["timestamp"], utc=True).astype("int64").to_numpy() // 10**6
    return None


def missing_candles(df: pd.DataFrame, interval: str, last_n: Optional[int] = None) -> int:
    """Number of candles absent between the first and last bar (optionally only the last ``last_n`` bars)."""
    t = _open_times_ms(df)
    if t is None or len(t) < 2 or interval not in INTERVAL_SECONDS:
        return 0
    if last_n:
        t = t[-last_n:]
    step = INTERVAL_SECONDS[interval] * 1000
    gaps = np.diff(t) // step - 1
    return int(gaps[gaps > 0].sum())


def has_duplicates(df: pd.DataFrame) -> bool:
    t = _open_times_ms(df)
    return t is not None and len(np.unique(t)) != len(t)
