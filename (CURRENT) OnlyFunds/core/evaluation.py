"""Strategy evaluation across datasets / market regimes.

Answers "does anything have an edge?" with the metrics that matter after
costs: net return vs buy-and-hold, max drawdown, profit factor, expectancy,
Sharpe/Sortino, trade frequency, and out-of-sample walk-forward results.
A strategy is only called ``credible`` when its walk-forward out-of-sample
result is positive after costs on most folds, beats buy-and-hold, has
enough trades, and its profit factor exceeds 1.1.
"""
from __future__ import annotations

import copy
from typing import Dict, Iterable, List, Optional

import pandas as pd

from core.autotune import walk_forward
from core.backtester import BacktestConfig, run_backtest

EVAL_GRID = {"threshold": [0.25, 0.35, 0.45], "exit_rules.stop_atr_mult": [1.0, 1.5, 2.0]}
COLUMNS = ["dataset", "strategy", "trades", "trades_per_day", "total_return_pct", "buy_and_hold_return_pct",
           "max_drawdown_pct", "profit_factor", "expectancy", "win_rate_pct", "sharpe", "sortino", "fees_paid"]


def evaluate(frames: Dict[str, pd.DataFrame], base: Optional[BacktestConfig] = None,
             strategies: Iterable[str] = ("auto", "trend_following", "mean_reversion", "stacked"),
             walk_forward_folds: int = 4, grid: Optional[Dict] = None) -> Dict[str, pd.DataFrame]:
    base = base or BacktestConfig()
    grid = grid or EVAL_GRID
    fixed_rows: List[Dict] = []
    wf_rows: List[Dict] = []
    for name, df in frames.items():
        for strat in strategies:
            cfg = copy.deepcopy(base)
            cfg.strategy = strat
            m = run_backtest(df, cfg, name).metrics
            fixed_rows.append({"dataset": name, "strategy": strat, **{k: m[k] for k in COLUMNS[2:]}})
            if walk_forward_folds:
                wf = walk_forward(df, cfg, grid=grid, n_folds=walk_forward_folds, pair=name)
                f = wf["folds"]
                wf_rows.append({
                    "dataset": name, "strategy": strat,
                    "oos_compounded_return_pct": wf["oos_compounded_return_pct"],
                    "oos_buy_hold_compounded_pct": float(((1 + f.get("oos_buy_hold_pct", pd.Series([0.0])).fillna(0)
                                                           / 100).prod() - 1) * 100),
                    "profitable_folds": wf["profitable_folds"], "folds": wf["n_folds"],
                    "oos_trades": int(f["oos_trades"].sum()),
                })
    fixed = pd.DataFrame(fixed_rows, columns=COLUMNS)
    wf_df = pd.DataFrame(wf_rows)
    return {"fixed": fixed, "walk_forward": wf_df, "verdict": verdict(fixed, wf_df)}


def verdict(fixed: pd.DataFrame, wf: pd.DataFrame, min_trades: int = 30) -> pd.DataFrame:
    rows = []
    for (ds, st), g in fixed.groupby(["dataset", "strategy"]):
        r = g.iloc[0]
        w = wf[(wf["dataset"] == ds) & (wf["strategy"] == st)] if len(wf) else pd.DataFrame()
        reasons = []
        if r["trades"] < min_trades:
            reasons.append(f"only {int(r['trades'])} trades")
        if r["profit_factor"] <= 1.1:
            reasons.append(f"profit factor {r['profit_factor']:.2f}")
        if r["total_return_pct"] <= r["buy_and_hold_return_pct"]:
            reasons.append("does not beat buy-and-hold")
        if len(w):
            w0 = w.iloc[0]
            if w0["oos_compounded_return_pct"] <= 0:
                reasons.append("walk-forward OOS not profitable")
            if w0["profitable_folds"] <= w0["folds"] / 2:
                reasons.append(f"{int(w0['profitable_folds'])}/{int(w0['folds'])} profitable folds")
            if w0["oos_compounded_return_pct"] <= w0["oos_buy_hold_compounded_pct"]:
                reasons.append("walk-forward OOS does not beat buy-and-hold")
        rows.append({"dataset": ds, "strategy": st, "credible_edge": not reasons,
                     "reasons": "; ".join(reasons) or "passes all checks"})
    return pd.DataFrame(rows)
