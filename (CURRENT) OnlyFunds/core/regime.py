"""Market regime detection for monitoring and no-trade decisions (does NOT change strategy signals).

Causal: the label at bar t uses only bars <= t. Thresholds are fixed constants chosen a priori,
not tuned on any results. The detector may answer "uncertain".

Dimensions
- trend:      strong_up | weak_up | range | weak_down | strong_down   (from the causal trend score)
- volatility: low | normal | high                                     (rolling percentile of ATR %)
- vol_change: expanding | contracting | stable                        (ATR % vs its value k bars ago)
- confidence: 0..1, distance of the inputs from the nearest decision threshold
- label:      "<trend>/<volatility>[/<vol_change>]" or "uncertain" when confidence < min_confidence

Evaluation helpers compare the causal labels with an ex-post (non-causal, evaluation-only) trend
label and measure stability, false (reverting) transitions and detection delay.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from core.indicators import add_indicators

TREND_CLASSES = ("strong_down", "weak_down", "range", "weak_up", "strong_up")


@dataclass(frozen=True)
class RegimeParams:
    weak_trend: float = 0.25
    strong_trend: float = 0.60
    vol_window: int = 500
    vol_low: float = 0.25
    vol_high: float = 0.75
    vol_change_lag: int = 24
    expanding: float = 1.25
    contracting: float = 0.80
    margin: float = 0.08          # distance from a threshold that counts as fully confident
    min_confidence: float = 0.25  # below this the label is "uncertain"
    persistence: int = 1          # bars a new trend class must persist before it is adopted (1 = off)


def _trend_class(ts: np.ndarray, p: RegimeParams) -> np.ndarray:
    out = np.full(len(ts), None, dtype=object)
    ok = np.isfinite(ts)
    out[ok & (ts >= p.strong_trend)] = "strong_up"
    out[ok & (ts >= p.weak_trend) & (ts < p.strong_trend)] = "weak_up"
    out[ok & (np.abs(ts) < p.weak_trend)] = "range"
    out[ok & (ts <= -p.weak_trend) & (ts > -p.strong_trend)] = "weak_down"
    out[ok & (ts <= -p.strong_trend)] = "strong_down"
    return out


def _apply_persistence(cls: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return cls
    out = cls.copy()
    current, candidate, count = None, None, 0
    for i, c in enumerate(cls):
        if c is None:
            out[i] = current
            continue
        if current is None:
            current = c
        elif c != current:
            if c == candidate:
                count += 1
            else:
                candidate, count = c, 1
            if count >= n:
                current, candidate, count = c, None, 0
        else:
            candidate, count = None, 0
        out[i] = current
    return out


def detect(df: pd.DataFrame, params: Optional[RegimeParams] = None) -> pd.DataFrame:
    p = params or RegimeParams()
    ind = df if "trend_score" in df.columns else add_indicators(df)
    ts = ind["trend_score"].to_numpy(float)
    atrp = ind["atr_pct"]
    vol_pct = atrp.rolling(p.vol_window, min_periods=max(50, p.vol_window // 5)).rank(pct=True).to_numpy(float)
    vol_ratio = (atrp / atrp.shift(p.vol_change_lag)).to_numpy(float)
    trend = _apply_persistence(_trend_class(ts, p), p.persistence)
    vol = np.full(len(ts), None, dtype=object)
    okv = np.isfinite(vol_pct)
    vol[okv & (vol_pct < p.vol_low)] = "low"
    vol[okv & (vol_pct >= p.vol_low) & (vol_pct <= p.vol_high)] = "normal"
    vol[okv & (vol_pct > p.vol_high)] = "high"
    chg = np.where(vol_ratio > p.expanding, "expanding", np.where(vol_ratio < p.contracting, "contracting", "stable"))
    chg = np.where(np.isfinite(vol_ratio), chg, None)
    # confidence: closeness of the trend score and vol percentile to their decision thresholds
    t_thr = np.array([-p.strong_trend, -p.weak_trend, p.weak_trend, p.strong_trend])
    d_t = np.min(np.abs(ts[:, None] - t_thr[None, :]), axis=1)
    v_thr = np.array([p.vol_low, p.vol_high])
    d_v = np.min(np.abs(vol_pct[:, None] - v_thr[None, :]), axis=1)
    conf = np.clip(np.minimum(d_t, d_v) / p.margin, 0, 1)
    valid = np.array([t is not None and v is not None and c is not None for t, v, c in zip(trend, vol, chg)])
    label = np.where(valid & (conf >= p.min_confidence),
                     np.array([f"{t}/{v}/{c}" for t, v, c in zip(trend, vol, chg)], dtype=object), "uncertain")
    label = np.where(valid, label, None)
    return pd.DataFrame({"trend": trend, "volatility": vol, "vol_change": chg, "confidence": np.where(valid, conf, np.nan),
                         "label": label, "trend_score": ts, "vol_percentile": vol_pct}, index=df.index)


# ------------------------------------------------------------------ evaluation (uses future data!)
def expost_trend(close: pd.Series, window: int = 48, strong: float = 2.0, weak: float = 0.75) -> pd.Series:
    """NON-CAUSAL reference label: forward return over ``window`` bars in units of its own volatility.
    For evaluation only, never for decisions."""
    fwd = np.log(close.shift(-window) / close)
    vol = np.log(close).diff().rolling(window, min_periods=window).std() * np.sqrt(window)
    z = fwd / vol
    out = pd.Series(None, index=close.index, dtype=object)
    out[z >= strong] = "strong_up"
    out[(z >= weak) & (z < strong)] = "weak_up"
    out[z.abs() < weak] = "range"
    out[(z <= -weak) & (z > -strong)] = "weak_down"
    out[z <= -strong] = "strong_down"
    return out


def trailing_trend(close: pd.Series, window: int = 48, strong: float = 2.0, weak: float = 0.75) -> pd.Series:
    """Causal counterpart of expost_trend: the PAST ``window``-bar return in volatility units.
    Agreement with it measures how well the detector describes the recent past."""
    back = np.log(close / close.shift(window))
    vol = np.log(close).diff().rolling(window, min_periods=window).std() * np.sqrt(window)
    z = back / vol
    out = pd.Series(None, index=close.index, dtype=object)
    out[z >= strong] = "strong_up"
    out[(z >= weak) & (z < strong)] = "weak_up"
    out[z.abs() < weak] = "range"
    out[(z <= -weak) & (z > -strong)] = "weak_down"
    out[z <= -strong] = "strong_down"
    return out


def _direction(cls: pd.Series) -> pd.Series:
    m = {"strong_up": "up", "weak_up": "up", "range": "range", "weak_down": "down", "strong_down": "down"}
    return cls.map(m)


def evaluate(det: pd.DataFrame, close: pd.Series, revert_within: int = 6, horizon: int = 48) -> Dict[str, float]:
    trend = det["trend"]
    valid = trend.notna()
    t = trend[valid]
    changes = (t != t.shift()).iloc[1:]
    n_trans = int(changes.sum())
    runs = (t != t.shift()).cumsum()
    run_len = t.groupby(runs).size()
    # false transitions: a change that reverts to the previous class within `revert_within` bars
    vals = t.to_numpy()
    idx = np.where(vals[1:] != vals[:-1])[0] + 1
    reverts = sum(1 for i in idx if i >= 1 and any(vals[j] == vals[i - 1] for j in range(i + 1, min(len(vals), i + 1 + revert_within))))
    # agreement with the ex-post forward trend (direction level: up / range / down)
    truth = expost_trend(close, horizon)
    both = valid & truth.notna()
    agree = float((_direction(trend[both]) == _direction(truth[both])).mean()) if both.any() else float("nan")
    # chance level: agreement if labels were independent with the same marginals
    pd_, pt_ = _direction(trend[both]).value_counts(normalize=True), _direction(truth[both]).value_counts(normalize=True)
    chance = float(sum(pd_.get(k, 0) * pt_.get(k, 0) for k in ("up", "range", "down")))
    # detection delay: bars from an ex-post direction change until the detector shows that direction
    td = _direction(truth).to_numpy()
    dd = _direction(trend).to_numpy()
    delays = []
    for i in range(1, len(td)):
        if td[i] is not None and td[i - 1] is not None and td[i] != td[i - 1] and td[i] != "range":
            for k in range(0, 200):
                if i + k < len(dd) and dd[i + k] == td[i]:
                    delays.append(k)
                    break
    past = trailing_trend(close, horizon)
    bp = valid & past.notna()
    agree_past = float((_direction(trend[bp]) == _direction(past[bp])).mean()) if bp.any() else float("nan")
    return {"bars": int(valid.sum()), "transitions_per_1000_bars": n_trans / max(len(t), 1) * 1000,
            "agreement_with_trailing_direction": agree_past,
            "median_run_bars": float(run_len.median()) if len(run_len) else float("nan"),
            "false_transition_share": reverts / max(n_trans, 1),
            "agreement_with_expost_direction": agree, "chance_agreement": chance,
            "median_detection_delay_bars": float(np.median(delays)) if delays else float("nan"),
            "uncertain_share": float((det["label"] == "uncertain").mean())}
