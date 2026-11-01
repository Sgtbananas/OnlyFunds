"""Parameter optimisation with out-of-sample validation.

Parameters are chosen on the training segment only, then evaluated once on
an unseen test segment. ``walk_forward`` repeats this over rolling windows.
A result is only flagged ``confirmed`` when the out-of-sample run is
profitable after fees, has enough trades, and beats buy-and-hold over the
same segment. Nothing is applied automatically.
"""
from __future__ import annotations

import copy
import itertools
import logging
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from core.backtester import BacktestConfig, BacktestResult, run_backtest
from core.indicators import warmup_bars

logger = logging.getLogger(__name__)

DEFAULT_GRID: Dict[str, List[Any]] = {
    "threshold": [0.25, 0.35, 0.45],
    "exit_rules.stop_atr_mult": [1.0, 1.5, 2.0],
    "exit_rules.tp_atr_mult": [2.0, 3.0],
    "strategy": ["auto", "trend_following", "mean_reversion"],
}


def _apply(cfg: BacktestConfig, params: Dict[str, Any]) -> BacktestConfig:
    c = copy.deepcopy(cfg)
    for key, val in params.items():
        obj = c
        parts = key.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        if parts[0] in ("indicator_params", "signal_params"):
            getattr(c, parts[0])[parts[1]] = val
        else:
            setattr(obj, parts[-1], val)
    return c


def _score(res: BacktestResult, objective: str, min_trades: int) -> float:
    m = res.metrics
    if m["trades"] < min_trades:
        return -np.inf
    if objective == "sharpe":
        return m["sharpe"]
    if objective == "return":
        return m["total_return_pct"]
    if objective == "return_over_dd":
        return m["total_return_pct"] / max(m["max_drawdown_pct"], 1.0)
    raise ValueError(f"unknown objective {objective}")


def grid_combinations(grid: Dict[str, Iterable[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(list(grid[k]) for k in keys))]


def _segment_with_warmup(df: pd.DataFrame, start: int, end: int, warm: int) -> pd.DataFrame:
    return df.iloc[max(0, start - warm):end].reset_index(drop=True)


def optimise(df: pd.DataFrame, base: Optional[BacktestConfig] = None, grid: Optional[Dict] = None,
             train_frac: float = 0.7, objective: str = "return_over_dd", min_trades: int = 8,
             pair: str = "PAIR") -> Dict[str, Any]:
    base = base or BacktestConfig()
    grid = grid or DEFAULT_GRID
    cut = int(len(df) * train_frac)
    warm = warmup_bars(base.indicator_params)
    train = df.iloc[:cut].reset_index(drop=True)
    test = _segment_with_warmup(df, cut, len(df), warm)
    rows = []
    best, best_score = None, -np.inf
    for params in grid_combinations(grid):
        res = run_backtest(train, _apply(base, params), pair)
        sc = _score(res, objective, min_trades)
        rows.append({**params, "score": sc, "trades": res.metrics["trades"],
                     "return_pct": res.metrics["total_return_pct"], "max_dd_pct": res.metrics["max_drawdown_pct"]})
        if sc > best_score:
            best, best_score = params, sc
    table = pd.DataFrame(rows).sort_values("score", ascending=False)
    if best is None:
        return {"best_params": None, "confirmed": False, "table": table,
                "reason": f"no parameter set reached {min_trades} trades in-sample"}
    best_cfg = _apply(base, best)
    ins = run_backtest(train, best_cfg, pair)
    oos = run_backtest(test, best_cfg, pair)
    confirmed, reason = _confirm(oos, min_trades=max(3, min_trades // 3))
    return {"best_params": best, "in_sample": ins.metrics, "out_of_sample": oos.metrics,
            "confirmed": confirmed, "reason": reason, "table": table, "best_config": best_cfg}


def _confirm(oos: BacktestResult, min_trades: int) -> (bool, str):
    m = oos.metrics
    if m["trades"] < min_trades:
        return False, f"only {m['trades']} out-of-sample trades"
    if m["total_return_pct"] <= 0:
        return False, f"out-of-sample return {m['total_return_pct']:.2f}% is not positive after fees"
    if m["total_return_pct"] <= m["buy_and_hold_return_pct"]:
        return False, "out-of-sample return does not beat buy-and-hold"
    return True, "profitable out-of-sample and beats buy-and-hold"


def walk_forward(df: pd.DataFrame, base: Optional[BacktestConfig] = None, grid: Optional[Dict] = None,
                 n_folds: int = 4, train_bars: Optional[int] = None, objective: str = "return_over_dd",
                 min_trades: int = 5, pair: str = "PAIR") -> Dict[str, Any]:
    """Rolling train -> test folds. Test windows never overlap and never feed training."""
    base = base or BacktestConfig()
    grid = grid or DEFAULT_GRID
    warm = warmup_bars(base.indicator_params)
    n = len(df)
    train_bars = train_bars or n // (n_folds + 1)
    test_bars = (n - train_bars) // n_folds
    if test_bars < warm:
        raise ValueError("not enough data for walk-forward with these settings")
    folds = []
    for k in range(n_folds):
        tr_start = k * test_bars
        tr_end = tr_start + train_bars
        te_end = min(n, tr_end + test_bars)
        train = df.iloc[tr_start:tr_end].reset_index(drop=True)
        best, best_score = None, -np.inf
        for params in grid_combinations(grid):
            sc = _score(run_backtest(train, _apply(base, params), pair), objective, min_trades)
            if sc > best_score:
                best, best_score = params, sc
        if best is None:
            folds.append({"fold": k, "params": None, "oos_return_pct": 0.0, "oos_trades": 0,
                          "note": "no params met min_trades; stood aside"})
            continue
        oos = run_backtest(_segment_with_warmup(df, tr_end, te_end, warm), _apply(base, best), pair)
        folds.append({"fold": k, "params": best, "oos_return_pct": oos.metrics["total_return_pct"],
                      "oos_trades": oos.metrics["trades"], "oos_max_dd_pct": oos.metrics["max_drawdown_pct"],
                      "oos_buy_hold_pct": oos.metrics["buy_and_hold_return_pct"]})
    f = pd.DataFrame(folds)
    compounded = float((np.prod(1 + f["oos_return_pct"].fillna(0) / 100) - 1) * 100)
    return {"folds": f, "oos_compounded_return_pct": compounded,
            "profitable_folds": int((f["oos_return_pct"] > 0).sum()), "n_folds": n_folds}


def ml_out_of_sample_backtest(df: pd.DataFrame, cfg: BacktestConfig, pair: str = "PAIR",
                              train_frac: float = 0.7, horizon: int = 48) -> Dict[str, Any]:
    """Train the ML filter on the first part only, then compare rule-only vs ML-gated on the rest."""
    from core.core_signals import generate_signals
    from core.ml_filter import predict_proba, train_filter

    cut = int(len(df) * train_frac)
    warm = warmup_bars(cfg.indicator_params)
    bundle = train_filter({pair: df.iloc[:cut].reset_index(drop=True)}, cfg.exit_rules, horizon=horizon,
                          threshold=cfg.ml_threshold, strategy=cfg.strategy, signal_threshold=cfg.threshold,
                          save_path=None)
    test = _segment_with_warmup(df, cut, len(df), warm)
    sig = generate_signals(test, strategy=cfg.strategy, threshold=cfg.threshold, params=cfg.signal_params,
                           indicator_params=cfg.indicator_params)
    proba = predict_proba(bundle, sig)
    rule_only = run_backtest(test, cfg, pair, signals=sig)
    gated = run_backtest(test, cfg, pair, ml_proba=proba, signals=sig)
    return {"report": bundle["report"], "rule_only": rule_only.metrics, "ml_gated": gated.metrics}


def suggest_tuning_parameters(metrics: Dict[str, float]) -> List[str]:
    """Plain-language hints from backtest metrics (``run_backtest(...).metrics``)."""
    s = []
    if metrics.get("trades", 0) < 10:
        s.append("Few trades: results are not statistically meaningful yet.")
    if metrics.get("win_rate_pct", 0) < 40:
        s.append("Win rate below 40%: consider a higher entry threshold or wider stops.")
    if metrics.get("max_drawdown_pct", 0) > 10:
        s.append("Drawdown above 10%: reduce risk per trade or tighten exposure limits.")
    if metrics.get("fees_paid", 0) > abs(metrics.get("realised_pnl", 0)) and metrics.get("trades", 0):
        s.append("Fees exceed net PnL: trade less often or use a longer interval.")
    if metrics.get("total_return_pct", 0) < metrics.get("buy_and_hold_return_pct", 0):
        s.append("Strategy underperforms buy-and-hold on this data.")
    return s or ["No obvious problems in these metrics."]
