"""ML entry filter (meta-model on top of the rule-based signals).

What it predicts: for a bar where the rule engine wants to enter, will a
long entered at the next open hit its take-profit before its stop-loss
within ``horizon`` bars (triple-barrier label, same ATR multipliers as the
exit rules)?

Leakage controls:
- Features are causal indicator values at bar t (see ``FEATURES``).
- Labels use bars t+1 .. t+horizon only, and only for training.
- Train / validation split is chronological per pair, with a purge gap of
  ``horizon`` bars so no training label overlaps validation features.

The model is only marked ``useful`` if, on the held-out validation period:
- ROC AUC >= 0.55, and
- the trades it accepts have a positive expectancy in R-multiples *after
  fees and stop slippage*, higher than taking every candidate entry.
Win rate alone is never enough: a filter can raise win rate while cutting
profit. Otherwise the app reports "no demonstrated edge" and does not gate
entries with it by default.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from core.core_signals import component_signals, generate_signals
from core.positions import ExitRules

logger = logging.getLogger(__name__)

APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(APP_ROOT, "state", "ml_filter.pkl")
FEATURES = ["rsi", "ema_diff", "macd_hist_n", "bb_pctb", "bb_width", "atr_pct", "volatility", "trend_score",
            "ret_1", "ret_3", "ret_12", "vol_ratio", "ema_cross", "rsi_reversion", "macd_momentum",
            "bollinger_reversion", "signal", "confidence"]
MODEL_VERSION = 3


def build_features(sig: pd.DataFrame) -> pd.DataFrame:
    """Scale-free causal features from a ``generate_signals`` frame."""
    f = pd.DataFrame(index=sig.index)
    for c in ("rsi", "ema_diff", "bb_pctb", "bb_width", "atr_pct", "volatility", "trend_score", "signal",
              "confidence"):
        f[c] = sig[c]
    f["macd_hist_n"] = sig["macd_hist"] / sig["Close"]
    close = sig["Close"]
    f["ret_1"] = close.pct_change(1)
    f["ret_3"] = close.pct_change(3)
    f["ret_12"] = close.pct_change(12)
    f["vol_ratio"] = sig["volatility"] / sig["volatility"].rolling(100, min_periods=20).mean()
    comps = component_signals(sig)
    for c in ("ema_cross", "rsi_reversion", "macd_momentum", "bollinger_reversion"):
        f[c] = comps[c]
    return f[FEATURES].replace([np.inf, -np.inf], np.nan)


def triple_barrier_labels(df: pd.DataFrame, atr: pd.Series, rules: ExitRules, horizon: int = 48) -> pd.Series:
    """1 = take-profit hit first, 0 = stop hit first or neither within horizon. NaN where unknown.

    Entry is the open of bar t+1. A bar touching both barriers counts as a stop (conservative).
    """
    o, h, l = (df[c].to_numpy(float) for c in ("Open", "High", "Low"))
    a = atr.to_numpy(float)
    n = len(df)
    out = np.full(n, np.nan)
    for t in range(n - 1):
        if not np.isfinite(a[t]) or a[t] <= 0:
            continue
        end = t + 1 + horizon
        if end > n:
            break
        entry = o[t + 1]
        stop, tp = entry - rules.stop_atr_mult * a[t], entry + rules.tp_atr_mult * a[t]
        label = 0.0
        for j in range(t + 1, end):
            if l[j] <= stop:
                label = 0.0
                break
            if h[j] >= tp:
                label = 1.0
                break
        out[t] = label
    return pd.Series(out, index=df.index)


def barrier_outcomes(df: pd.DataFrame, atr: pd.Series, rules: ExitRules, horizon: int = 48,
                     fee_rate: float = 0.002, stop_slippage_pct: float = 0.002) -> pd.Series:
    """Net outcome of a long entered at open[t+1], in R (1R = stop distance), after costs.

    take-profit -> +tp/stop R; stop -> -1R minus slippage; neither -> close at the horizon.
    Costs (two fees, stop slippage) are converted to R and subtracted. NaN where unknown.
    """
    o, h, l, c = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    a = atr.to_numpy(float)
    n = len(df)
    out = np.full(n, np.nan)
    for t in range(n - 1):
        end = t + 1 + horizon
        if end > n or not np.isfinite(a[t]) or a[t] <= 0:
            continue
        entry = o[t + 1]
        risk = rules.stop_atr_mult * a[t]
        stop, tp = entry - risk, entry + rules.tp_atr_mult * a[t]
        exit_px = c[end - 1]
        for j in range(t + 1, end):   # same order as core.positions.check_exit
            if o[j] <= stop:
                exit_px = o[j] * (1 - stop_slippage_pct)
                break
            if o[j] >= tp:
                exit_px = o[j]
                break
            if l[j] <= stop:
                exit_px = stop * (1 - stop_slippage_pct)
                break
            if h[j] >= tp:
                exit_px = tp
                break
        net = exit_px * (1 - fee_rate) - entry * (1 + fee_rate)
        out[t] = net / risk
    return pd.Series(out, index=df.index)


def build_dataset(frames: Dict[str, pd.DataFrame], rules: ExitRules, horizon: int = 48,
                  strategy: str = "auto", threshold: float = 0.35, val_frac: float = 0.3,
                  candidates_only: bool = True, return_r: bool = False, fee_rate: float = 0.002,
                  stop_slippage_pct: float = 0.002):
    """Chronological per-pair split with a purge gap. Returns (Xtr, ytr, Xva, yva)."""
    tr_X, tr_y, va_X, va_y, va_r = [], [], [], [], []
    for pair, df in frames.items():
        sig = generate_signals(df.reset_index(drop=True), strategy=strategy, threshold=threshold)
        X = build_features(sig)
        y = triple_barrier_labels(sig, sig["ATR"], rules, horizon)
        r = barrier_outcomes(sig, sig["ATR"], rules, horizon, fee_rate, stop_slippage_pct)
        mask = y.notna() & X.notna().all(axis=1)
        if candidates_only:
            mask &= sig["entry"]
        cut = int(len(sig) * (1 - val_frac))
        idx = np.arange(len(sig))
        train_m = mask & (idx < cut - horizon)        # purge: labels must end before validation starts
        val_m = mask & (idx >= cut)
        tr_X.append(X[train_m]); tr_y.append(y[train_m])
        va_X.append(X[val_m]); va_y.append(y[val_m]); va_r.append(r[val_m])
    cat = lambda parts: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if return_r:
        return cat(tr_X), cat(tr_y), cat(va_X), cat(va_y), cat(va_r)
    return cat(tr_X), cat(tr_y), cat(va_X), cat(va_y)


def train_filter(frames: Dict[str, pd.DataFrame], rules: Optional[ExitRules] = None, horizon: int = 48,
                 threshold: float = 0.55, strategy: str = "auto", signal_threshold: float = 0.35,
                 min_samples: int = 150, save_path: Optional[str] = MODEL_PATH, fee_rate: float = 0.002,
                 stop_slippage_pct: float = 0.002) -> Dict:
    """Train and validate the filter. Returns the model bundle (also saved when ``save_path``)."""
    rules = rules or ExitRules()
    kw = dict(return_r=True, fee_rate=fee_rate, stop_slippage_pct=stop_slippage_pct)
    Xtr, ytr, Xva, yva, rva = build_dataset(frames, rules, horizon, strategy, signal_threshold, **kw)
    candidates_only = True
    if len(Xtr) < min_samples:
        # Too few rule-based entries: learn from all bars instead (documented in the report).
        Xtr, ytr, Xva, yva, rva = build_dataset(frames, rules, horizon, strategy, signal_threshold,
                                                candidates_only=False, **kw)
        candidates_only = False
    if len(Xtr) < min_samples or len(Xva) < 30:
        raise ValueError(f"not enough labelled samples (train={len(Xtr)}, validation={len(Xva)}); "
                         "fetch more history")
    if ytr.nunique() < 2:
        raise ValueError("training labels have a single class; cannot train")
    model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                           l2_regularization=1.0, min_samples_leaf=40, random_state=42)
    model.fit(Xtr, ytr.astype(int))
    p_va = model.predict_proba(Xva)[:, 1]
    p_tr = model.predict_proba(Xtr)[:, 1]
    base_rate = float(yva.mean())
    auc = float(roc_auc_score(yva, p_va)) if yva.nunique() == 2 else float("nan")
    accepted = p_va >= threshold
    precision = float(yva[accepted].mean()) if accepted.any() else float("nan")
    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(), "version": MODEL_VERSION,
        "pairs": list(frames), "horizon": horizon, "threshold": threshold,
        "candidates_only": candidates_only,
        "train_samples": int(len(Xtr)), "validation_samples": int(len(Xva)),
        "train_auc": float(roc_auc_score(ytr, p_tr)),
        "validation_auc": auc, "validation_base_win_rate": base_rate,
        "validation_precision_at_threshold": precision,
        "validation_accept_rate": float(accepted.mean()),
        "exit_rules": rules.__dict__,
    }
    base_r = float(rva.mean()) if len(rva) else float("nan")
    gated_r = float(rva[accepted].mean()) if accepted.any() else float("nan")
    report.update({
        "validation_expectancy_r_all": base_r,          # take every candidate
        "validation_expectancy_r_accepted": gated_r,    # take only what the model accepts
        "validation_accepted_trades": int(accepted.sum()),
        "costs": {"fee_rate": fee_rate, "stop_slippage_pct": stop_slippage_pct},
    })
    report["useful"] = edge_check(auc, int(accepted.sum()), gated_r, base_r)
    bundle = {"model": model, "features": FEATURES, "report": report}
    if save_path:
        save_bundle(bundle, save_path)
    logger.info("ML filter trained: %s", report)
    return bundle


def edge_check(auc: float, accepted_trades: int, expectancy_r_accepted: float, expectancy_r_all: float,
               min_auc: float = 0.55, min_trades: int = 10) -> bool:
    """The filter is useful only if it ranks well AND the trades it keeps make more money after costs."""
    vals = (auc, expectancy_r_accepted, expectancy_r_all)
    if not all(isinstance(v, (int, float)) and np.isfinite(v) for v in vals):
        return False
    return bool(auc >= min_auc and accepted_trades >= min_trades and expectancy_r_accepted > 0
                and expectancy_r_accepted > expectancy_r_all)


def save_bundle(bundle: Dict, path: str = MODEL_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    joblib.dump(bundle, tmp)
    os.replace(tmp, path)
    with open(os.path.splitext(path)[0] + "_report.json", "w", encoding="utf-8") as f:
        json.dump(bundle["report"], f, indent=2, default=str)


def load_bundle(path: str = MODEL_PATH) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    try:
        bundle = joblib.load(path)
    except Exception as exc:
        logger.error("could not load ML filter %s: %s", path, exc)
        return None
    if not isinstance(bundle, dict) or bundle.get("report", {}).get("version") != MODEL_VERSION:
        logger.warning("ignoring ML filter at %s: incompatible version", path)
        return None
    return bundle


def predict_proba(bundle: Dict, sig: pd.DataFrame) -> pd.Series:
    """Probability series aligned to ``sig`` (NaN where features are undefined)."""
    X = build_features(sig)
    ok = X.notna().all(axis=1)
    out = pd.Series(np.nan, index=sig.index)
    if ok.any():
        out[ok] = bundle["model"].predict_proba(X[ok][bundle["features"]])[:, 1]
    return out


def ml_confidence(df: pd.DataFrame, bundle: Optional[Dict] = None) -> Optional[float]:
    """Latest-bar probability, or None when no model / undefined features."""
    bundle = bundle or load_bundle()
    if bundle is None or df is None or df.empty:
        return None
    sig = generate_signals(df)
    p = predict_proba(bundle, sig).iloc[-1]
    return None if pd.isna(p) else float(p)
