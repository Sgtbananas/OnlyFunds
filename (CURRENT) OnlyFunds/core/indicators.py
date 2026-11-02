"""Technical indicators used by every OF2 strategy.

All indicators are *causal*: the value at bar ``t`` only uses data from bars
``<= t``. This is what keeps the backtester free of look-ahead bias.

Conventions follow the standard definitions:
- EMA: pandas ``ewm(span=n, adjust=False)`` (recursive EMA, alpha = 2/(n+1)).
- RSI / ATR: Wilder smoothing (alpha = 1/n), as defined by J. Welles Wilder.
- MACD: EMA(fast) - EMA(slow), signal = EMA(macd, signal), hist = macd - signal.
- Bollinger: SMA(n) +/- k * population std (ddof=0), as defined by J. Bollinger.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

DEFAULT_PARAMS: Dict[str, float] = {
    "ema_fast": 9,
    "ema_slow": 21,
    "ema_trend": 50,
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bb_window": 20,
    "bb_std": 2.0,
    "atr_period": 14,
    "vol_window": 20,
    "trend_window": 20,
}

REQUIRED_OHLC = ("Open", "High", "Low", "Close")


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = wilder_smooth(gain, period)
    avg_loss = wilder_smooth(loss, period)
    rs = avg_gain / avg_loss
    out = 100.0 - 100.0 / (1.0 + rs)
    # No losses in the window -> RSI 100; no movement at all -> 50.
    out = out.where(avg_loss != 0, 100.0)
    out = out.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    return out.where(avg_gain.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    if fast >= slow:
        raise ValueError("MACD fast period must be shorter than slow period")
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


def bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower) / mid
    pct_b = (close - lower) / (upper - lower)
    return mid, upper, lower, width, pct_b


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    tr.iloc[0] = high.iloc[0] - low.iloc[0]
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    return wilder_smooth(true_range(high, low, close), period)


def trend_score(close: pd.Series, ema_fast: pd.Series, ema_slow: pd.Series, ema_trend: pd.Series,
                window: int = 20) -> pd.Series:
    """Trend score in [-1, 1].

    Average of three bounded components:
    - EMA alignment (fast vs slow vs trend EMA ordering)
    - slope of the trend EMA over ``window`` bars, scaled by volatility
    - price position relative to the trend EMA, scaled by volatility
    """
    align = (np.sign(ema_fast - ema_slow) + np.sign(ema_slow - ema_trend)) / 2.0
    vol = close.pct_change().rolling(window, min_periods=window).std()
    slope = ema_trend.pct_change(window) / (vol * np.sqrt(window))
    pos = (close / ema_trend - 1.0) / (vol * np.sqrt(window))
    score = (align + np.tanh(slope) + np.tanh(pos)) / 3.0
    return score.clip(-1.0, 1.0)


def validate_ohlcv(df: Optional[pd.DataFrame]) -> bool:
    if df is None or df.empty:
        return False
    if any(c not in df.columns for c in REQUIRED_OHLC):
        return False
    if (df[list(REQUIRED_OHLC)] <= 0).any().any():
        return False
    return True


def add_indicators(df: pd.DataFrame, params: Optional[Dict] = None) -> pd.DataFrame:
    """Return a copy of ``df`` with every indicator column the strategies use.

    Columns added: ema_fast, ema_slow, ema_trend, ema_diff, rsi, macd,
    macd_signal, macd_hist, bb_mid, bb_upper, bb_lower, bb_width, bb_pctb,
    ATR, atr_pct, volatility, trend_score, returns.
    """
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()
    if any(c not in df.columns for c in REQUIRED_OHLC):
        raise ValueError(f"OHLC columns required, got {list(df.columns)}")
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update({k: v for k, v in params.items() if k in DEFAULT_PARAMS})

    out = df.copy()
    close, high, low = out["Close"].astype(float), out["High"].astype(float), out["Low"].astype(float)

    out["ema_fast"] = ema(close, int(p["ema_fast"]))
    out["ema_slow"] = ema(close, int(p["ema_slow"]))
    out["ema_trend"] = ema(close, int(p["ema_trend"]))
    out["ema_diff"] = (out["ema_fast"] - out["ema_slow"]) / close
    out["rsi"] = rsi(close, int(p["rsi_period"]))
    out["macd"], out["macd_signal"], out["macd_hist"] = macd(
        close, int(p["macd_fast"]), int(p["macd_slow"]), int(p["macd_signal"]))
    (out["bb_mid"], out["bb_upper"], out["bb_lower"],
     out["bb_width"], out["bb_pctb"]) = bollinger(close, int(p["bb_window"]), float(p["bb_std"]))
    out["ATR"] = atr(high, low, close, int(p["atr_period"]))
    out["atr_pct"] = out["ATR"] / close
    out["returns"] = close.pct_change()
    out["volatility"] = out["returns"].rolling(int(p["vol_window"]), min_periods=int(p["vol_window"])).std()
    out["trend_score"] = trend_score(close, out["ema_fast"], out["ema_slow"], out["ema_trend"],
                                     int(p["trend_window"]))
    return out


def warmup_bars(params: Optional[Dict] = None) -> int:
    """Number of bars before every indicator is defined."""
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    return int(max(p["ema_trend"], p["macd_slow"] + p["macd_signal"], p["bb_window"],
                   p["atr_period"], p["vol_window"], p["trend_window"] * 2, p["rsi_period"])) + 1
