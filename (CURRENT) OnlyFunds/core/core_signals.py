"""Signal engine: indicator components -> strategy stacks -> entry/exit decisions.

Every value at bar ``t`` uses only data up to and including bar ``t``. The
backtester and paper trader act on a bar's signal at the *next* bar, so no
decision can see the price it trades at.

OF2 trades spot, long-only. A positive signal means "go/stay long", a
negative signal means "exit / stay flat". Entries and exits are mutually
exclusive by construction (entry needs signal >= +threshold, exit needs
signal <= -exit_threshold).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from core.indicators import add_indicators

COMPONENTS = ("ema_cross", "rsi_reversion", "macd_momentum", "bollinger_reversion", "trend_score")

STRATEGIES: Dict[str, Dict[str, float]] = {
    "trend_following": {"ema_cross": 0.35, "macd_momentum": 0.35, "trend_score": 0.30},
    "mean_reversion": {"rsi_reversion": 0.5, "bollinger_reversion": 0.5},
    # Trend filter + pullback: an equal-weight blend of all five never fires,
    # because trend and mean-reversion components cancel by construction.
    "stacked": {"trend_score": 0.4, "ema_cross": 0.2, "rsi_reversion": 0.2, "bollinger_reversion": 0.2},
}
STRATEGY_NAMES = tuple(STRATEGIES) + ("auto",)

REGIMES = ("trending_up", "trending_down", "ranging", "high_volatility")

DEFAULT_SIGNAL_PARAMS = {
    "rsi_low": 30.0,
    "rsi_high": 70.0,
    "trend_threshold": 0.35,     # |trend_score| above this = trending
    "high_vol_pct": 0.90,        # ATR% rolling percentile above this = high volatility
    "vol_rank_window": 200,
    "min_confirmations": 2,      # components that must agree with an entry
    "exit_threshold": 0.15,
}

# Regime -> strategy used by "auto". Trending-down maps to None: spot long-only
# has no edge there, so the engine stands aside (capital protection).
REGIME_STRATEGY = {
    "trending_up": "trend_following",
    "ranging": "mean_reversion",
    "high_volatility": "stacked",
    "trending_down": None,
}


def _p(params: Optional[Dict]) -> Dict:
    out = dict(DEFAULT_SIGNAL_PARAMS)
    if params:
        out.update({k: v for k, v in params.items() if k in DEFAULT_SIGNAL_PARAMS})
    return out


def component_signals(df: pd.DataFrame, params: Optional[Dict] = None) -> pd.DataFrame:
    """Each component scaled to [-1, 1]. Requires ``add_indicators`` columns."""
    p = _p(params)
    vol = df["volatility"].replace(0, np.nan)
    out = pd.DataFrame(index=df.index)
    out["ema_cross"] = np.tanh(df["ema_diff"] / (2.0 * vol))
    rsi = df["rsi"]
    lo, hi = float(p["rsi_low"]), float(p["rsi_high"])
    rsi_sig = pd.Series(0.0, index=df.index)
    rsi_sig = rsi_sig.where(~(rsi < lo), ((lo - rsi) / lo).clip(0, 1) * 0.5 + 0.5)
    rsi_sig = rsi_sig.where(~(rsi > hi), -(((rsi - hi) / (100 - hi)).clip(0, 1) * 0.5 + 0.5))
    out["rsi_reversion"] = rsi_sig.where(rsi.notna())
    out["macd_momentum"] = np.tanh((df["macd_hist"] / df["Close"]) / vol)
    out["bollinger_reversion"] = ((0.5 - df["bb_pctb"]) * 2.0).clip(-1, 1)
    out["trend_score"] = df["trend_score"]
    return out.clip(-1, 1)


def classify_regime(df: pd.DataFrame, params: Optional[Dict] = None) -> pd.Series:
    p = _p(params)
    window = int(p["vol_rank_window"])
    vol_rank = df["atr_pct"].rolling(window, min_periods=max(20, window // 4)).rank(pct=True)
    ts = df["trend_score"]
    regime = pd.Series("ranging", index=df.index, dtype=object)
    regime[ts >= p["trend_threshold"]] = "trending_up"
    regime[ts <= -p["trend_threshold"]] = "trending_down"
    regime[vol_rank >= p["high_vol_pct"]] = "high_volatility"
    regime[ts.isna()] = None
    return regime


def stack(components: pd.DataFrame, strategy: str) -> pd.Series:
    weights = STRATEGIES[strategy]
    total = sum(weights.values())
    return sum(components[c] * w for c, w in weights.items()) / total


def confirmations(components: pd.DataFrame, strategy: str, direction: pd.Series, eps: float = 0.1) -> pd.Series:
    """How many of the strategy's components agree with ``direction`` (+1/-1)."""
    cols = list(STRATEGIES[strategy])
    agree = (np.sign(components[cols].where(components[cols].abs() > eps, 0)).mul(direction, axis=0) > 0)
    return agree.sum(axis=1)


def generate_signals(df: pd.DataFrame, strategy: str = "auto", threshold: float = 0.35,
                     params: Optional[Dict] = None, indicator_params: Optional[Dict] = None,
                     with_indicators: bool = True) -> pd.DataFrame:
    """Full decision frame.

    Columns: signal, confidence, regime, strategy, confirmations, entry, exit
    (plus every indicator column when ``with_indicators``).
    """
    if strategy not in STRATEGY_NAMES:
        raise ValueError(f"unknown strategy {strategy}; choose from {STRATEGY_NAMES}")
    p = _p(params)
    ind = df if "trend_score" in df.columns else add_indicators(df, indicator_params)
    comps = component_signals(ind, p)
    regime = classify_regime(ind, p)
    stacks = {name: stack(comps, name) for name in STRATEGIES}

    if strategy == "auto":
        chosen = regime.map(REGIME_STRATEGY)
    else:
        chosen = pd.Series(strategy, index=ind.index, dtype=object)

    signal = pd.Series(np.nan, index=ind.index)
    conf_count = pd.Series(0, index=ind.index)
    n_components = pd.Series(np.nan, index=ind.index)
    for name in STRATEGIES:
        mask = chosen == name
        if not mask.any():
            continue
        signal[mask] = stacks[name][mask]
        direction = np.sign(stacks[name]).replace(0, 1)
        conf_count[mask] = confirmations(comps, name, direction)[mask]
        n_components[mask] = len(STRATEGIES[name])

    # Trending-down regime under "auto": no new longs, and a mild exit bias via the trend stack.
    stand_aside = chosen.isna() & regime.notna()
    signal[stand_aside] = np.minimum(stacks["trend_following"][stand_aside], 0.0)

    confidence = (signal.abs() * (conf_count / n_components)).fillna(0.0).clip(0, 1)
    min_conf = min(int(p["min_confirmations"]), 99)
    need = np.minimum(min_conf, n_components.fillna(0))
    entry = (signal >= threshold) & (conf_count >= need) & chosen.notna()
    exit_ = signal <= -float(p["exit_threshold"])
    warm = comps.notna().all(axis=1) & regime.notna()
    entry &= warm
    exit_ &= warm

    out = ind.copy() if with_indicators else pd.DataFrame(index=ind.index)
    out["signal"] = signal.where(warm, 0.0).fillna(0.0)
    out["confidence"] = confidence.where(warm, 0.0)
    out["regime"] = regime
    out["strategy"] = chosen.where(~stand_aside, "stand_aside")
    out["confirmations"] = conf_count
    out["entry"] = entry.astype(bool)
    out["exit"] = exit_.astype(bool)
    return out


def generate_signal(df: pd.DataFrame, indicator_params: Optional[Dict] = None, strategy: str = "auto") -> pd.Series:
    """Composite signal series in [-1, 1] (kept for backwards compatibility)."""
    return generate_signals(df, strategy=strategy, indicator_params=indicator_params)["signal"]


def smooth_signal(signal: pd.Series, smoothing_window: int = 3) -> pd.Series:
    """Trailing (causal) moving average of a signal."""
    return signal.rolling(window=smoothing_window, min_periods=1).mean()
