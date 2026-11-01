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
                         "Low": low, "Close": close, "Volume": rng.uniform(1_000, 10_000, n)})


def _frame(close: np.ndarray, rng, wick: float, freq: str = "5min") -> pd.DataFrame:
    n = len(close)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, wick, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, wick, n)))
    ts = pd.date_range("2025-01-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"timestamp": ts, "created_at": ts.asi8 // 10**6, "Open": open_, "High": high,
                         "Low": low, "Close": close, "Volume": rng.uniform(1_000, 10_000, n)})


def synthetic_regime(kind: str, n: int = 4000, seed: int = 0, interval: str = "5m") -> pd.DataFrame:
    """Controlled market regimes for strategy evaluation (no claim of realism beyond the regime itself).

    trend_up / trend_down: drifting random walk; range: Ornstein-Uhlenbeck mean reversion around 100;
    high_vol: driftless walk with 3x volatility; random_walk: driftless; switching: blocks of all of them.
    """
    rng = np.random.default_rng(seed)
    k = {"5m": 1, "15m": 3, "30m": 6, "1h": 12, "4h": 48}[interval]  # time scale vs 5m
    vs, freq = np.sqrt(k), {"5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h", "4h": "4h"}[interval]
    if kind == "switching":
        kinds = ["trend_up", "range", "trend_down", "high_vol", "range", "trend_up"]
        parts, last = [], 100.0
        for k, kk in enumerate(kinds):
            seg = synthetic_regime(kk, n // len(kinds), seed + 100 + k, interval)["Close"].to_numpy()
            parts.append(seg / seg[0] * last)
            last = parts[-1][-1]
        return _frame(np.concatenate(parts), rng, 0.002 * vs, freq)
    if kind == "range":
        x = np.empty(n)
        x[0] = 100.0
        for i in range(1, n):
            x[i] = x[i - 1] + 0.02 * (100.0 - x[i - 1]) + rng.normal(0, 0.4 * vs)
        return _frame(np.maximum(x, 1.0), rng, 0.002 * vs, freq)
    vol = {"high_vol": 0.012}.get(kind, 0.004) * vs
    # Moderate, scale-consistent trend: drift = 3% of per-bar volatility (~5x over 4000 bars).
    drift = {"trend_up": 0.03, "trend_down": -0.03}.get(kind, 0.0) * vol
    if kind not in ("trend_up", "trend_down", "high_vol", "random_walk"):
        raise ValueError(kind)
    close = 100.0 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    return _frame(close, rng, vol / 2, freq)
