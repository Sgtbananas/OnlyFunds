"""Real-market validation per docs/VALIDATION_PROTOCOL.md.

Stage 1 (``run_dev_val``): tuning on DEV, out-of-sample checks on VAL, yearly walk-forward
2021-2024, ML trained on DEV and judged on VAL. Every decision for the final test is frozen.
Stage 2 (``run_test``): one evaluation of the frozen candidates on TEST.
"""
from __future__ import annotations

import copy
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.autotune import _apply, grid_combinations
from core.backtester import BacktestConfig, BacktestResult, run_backtest
from core.core_signals import generate_signals

PERIODS = {"DEV": ("2019-01-01", "2023-01-01"), "VAL": ("2023-01-01", "2025-01-01"),
           "TEST": ("2025-01-01", "2026-11-01")}
WALK_FORWARD_YEARS = (2021, 2022, 2023, 2024)
STRATEGIES = ("auto", "trend_following", "mean_reversion", "stacked")
GRID = {"threshold": [0.25, 0.35, 0.45], "exit_rules.stop_atr_mult": [1.0, 1.5, 2.0],
        "exit_rules.tp_atr_mult": [2.0, 3.0]}
HISTORY_BARS = 500           # indicator history before each trading window (no trading in it)
MIN_TRADES = 100
MIN_COIN_TRADES = 10
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 7


def protocol_fingerprint(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def segment(df: pd.DataFrame, start: str, end: str, history: int = HISTORY_BARS) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"], utc=True)
    s, e = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    first = int((ts < s).sum())
    out = df[(ts < e)].iloc[max(0, first - history):]
    return out.reset_index(drop=True)


def config_for(base: BacktestConfig, strategy: str, params: Optional[Dict[str, Any]], start: str) -> BacktestConfig:
    cfg = copy.deepcopy(base)
    cfg.strategy = strategy
    if params:
        cfg = _apply(cfg, params)
    cfg.trade_from = start
    return cfg


def run_period(data: Dict[str, pd.DataFrame], cfg_base: BacktestConfig, strategy: str,
               params: Optional[Dict[str, Any]], start: str, end: str,
               ml_bundle: Optional[Dict] = None) -> Dict[str, BacktestResult]:
    from core.ml_filter import predict_proba
    out = {}
    for pair, df in data.items():
        seg = segment(df, start, end)
        ts = pd.to_datetime(seg["timestamp"], utc=True)
        if seg.empty or (ts >= pd.Timestamp(start, tz="UTC")).sum() < 50:
            continue  # coin has no data in this window (e.g. delisted)
        cfg = config_for(cfg_base, strategy, params, start)
        sig = generate_signals(seg, strategy=cfg.strategy, threshold=cfg.threshold, params=cfg.signal_params,
                               indicator_params=cfg.indicator_params)
        proba = predict_proba(ml_bundle, sig) if ml_bundle is not None else None
        out[pair] = run_backtest(seg, cfg, pair, ml_proba=proba, signals=sig)
    return out


def bootstrap_lower(r: np.ndarray, n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> Tuple[float, float]:
    if len(r) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(r, size=(n, len(r)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def pooled(results: Dict[str, BacktestResult]) -> Dict[str, Any]:
    trades = pd.concat([r.trades.assign(pair=p) for p, r in results.items() if len(r.trades)], ignore_index=True) \
        if any(len(r.trades) for r in results.values()) else pd.DataFrame()
    per_coin = []
    for p, r in results.items():
        m = r.metrics
        per_coin.append({"pair": p, "trades": m["trades"], "net_return_pct": m["total_return_pct"],
                         "buy_hold_pct": m["buy_and_hold_return_pct"], "max_dd_pct": m["max_drawdown_pct"],
                         "profit_factor": m["profit_factor"], "expectancy_r": m["expectancy_r"],
                         "win_rate_pct": m["win_rate_pct"], "avg_win_r": m["avg_win_r"],
                         "avg_loss_r": m["avg_loss_r"], "fees": m["fees_paid"]})
    coins = pd.DataFrame(per_coin)
    out: Dict[str, Any] = {"per_coin": coins, "trades": int(len(trades))}
    if trades.empty:
        out.update({"expectancy_r": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                    "profit_factor": 0.0, "win_rate_pct": 0.0, "avg_win_r": 0.0, "avg_loss_r": 0.0,
                    "breadth": 0.0, "eligible_coins": 0, "exp_r_without_best_coin": float("nan"),
                    "by_regime": pd.DataFrame(), "by_year": pd.DataFrame(),
                    "mean_net_return_pct": float(coins["net_return_pct"].mean()) if len(coins) else 0.0,
                    "mean_buy_hold_pct": float(coins["buy_hold_pct"].mean()) if len(coins) else 0.0,
                    "worst_max_dd_pct": float(coins["max_dd_pct"].max()) if len(coins) else 0.0})
        return out
    r = trades["r_multiple"].to_numpy(float)
    pnl = trades["pnl"].to_numpy(float)
    gl = -pnl[pnl <= 0].sum()
    lo, hi = bootstrap_lower(r)
    eligible = coins[coins["trades"] >= MIN_COIN_TRADES]
    best = trades.groupby("pair")["r_multiple"].sum().idxmax()
    rest = trades[trades["pair"] != best]["r_multiple"]
    year = pd.to_datetime(trades["entry_time"], utc=True).dt.year
    out.update({
        "expectancy_r": float(r.mean()), "ci_low": lo, "ci_high": hi,
        "profit_factor": float(pnl[pnl > 0].sum() / gl) if gl > 0 else float("inf"),
        "win_rate_pct": float((pnl > 0).mean() * 100),
        "avg_win_r": float(r[pnl > 0].mean()) if (pnl > 0).any() else 0.0,
        "avg_loss_r": float(r[pnl <= 0].mean()) if (pnl <= 0).any() else 0.0,
        "eligible_coins": int(len(eligible)),
        "breadth": float((eligible["expectancy_r"] > 0).mean()) if len(eligible) else 0.0,
        "exp_r_without_best_coin": float(rest.mean()) if len(rest) else float("nan"),
        "by_regime": trades.groupby(trades["regime"].fillna("unknown"))["r_multiple"].agg(["count", "mean", "sum"])
        .rename(columns={"count": "trades", "mean": "expectancy_r", "sum": "total_r"}),
        "by_year": trades.groupby(year)["r_multiple"].agg(["count", "mean", "sum"])
        .rename(columns={"count": "trades", "mean": "expectancy_r", "sum": "total_r"}),
        "mean_net_return_pct": float(coins["net_return_pct"].mean()),
        "mean_buy_hold_pct": float(coins["buy_hold_pct"].mean()),
        "worst_max_dd_pct": float(coins["max_dd_pct"].max()),
    })
    return out


def criteria(p: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Protocol criteria 1-5 for one period. Returns (passed, failed reasons)."""
    fails = []
    if p["trades"] < MIN_TRADES:
        fails.append(f"{p['trades']} trades < {MIN_TRADES}")
    if not (p["expectancy_r"] > 0):
        fails.append(f"expectancy {p['expectancy_r']:+.3f}R ≤ 0")
    if not (p["ci_low"] > 0):
        fails.append(f"95% CI lower bound {p['ci_low']:+.3f}R ≤ 0")
    if not (p["profit_factor"] >= 1.10):
        fails.append(f"profit factor {p['profit_factor']:.2f} < 1.10")
    if not (p["breadth"] >= 0.60):
        fails.append(f"positive on {p['breadth']:.0%} of {p['eligible_coins']} coins < 60%")
    if not (p["exp_r_without_best_coin"] > 0):
        fails.append(f"without best coin {p['exp_r_without_best_coin']:+.3f}R ≤ 0")
    return (not fails), fails


def tune(data: Dict[str, pd.DataFrame], base: BacktestConfig, strategy: str, start: str, end: str
         ) -> Tuple[Optional[Dict[str, Any]], pd.DataFrame]:
    """Grid search on [start, end): max pooled expectancy in R with >= MIN_TRADES trades."""
    rows, best, best_score = [], None, -np.inf
    for params in grid_combinations(GRID):
        p = pooled(run_period(data, base, strategy, params, start, end))
        rows.append({**params, "trades": p["trades"], "expectancy_r": p["expectancy_r"]})
        if p["trades"] >= MIN_TRADES and p["expectancy_r"] > best_score:
            best, best_score = params, p["expectancy_r"]
    return best, pd.DataFrame(rows)


def walk_forward(data: Dict[str, pd.DataFrame], base: BacktestConfig, strategy: str,
                 years=WALK_FORWARD_YEARS) -> pd.DataFrame:
    rows = []
    for y in years:
        params, _ = tune(data, base, strategy, f"{y - 2}-01-01", f"{y}-01-01")
        p = pooled(run_period(data, base, strategy, params, f"{y}-01-01", f"{y + 1}-01-01")) if params else None
        rows.append({"year": y, "params": params, "trades": p["trades"] if p else 0,
                     "expectancy_r": p["expectancy_r"] if p else float("nan"),
                     "profit_factor": p["profit_factor"] if p else float("nan")})
    return pd.DataFrame(rows)


def walk_forward_pass(wf: pd.DataFrame) -> Tuple[bool, str]:
    pos = int((wf["expectancy_r"] > 0).sum())
    return pos >= 3, f"{pos}/{len(wf)} walk-forward years positive"


def train_ml(data: Dict[str, pd.DataFrame], base: BacktestConfig, strategy: str) -> Dict:
    from core.ml_filter import train_filter
    start, end = PERIODS["DEV"]
    frames = {p: segment(df, start, end, history=0) for p, df in data.items()}
    frames = {p: f for p, f in frames.items() if len(f) > 1000}
    return train_filter(frames, base.exit_rules, horizon=48, threshold=base.ml_threshold, strategy=strategy,
                        signal_threshold=base.threshold, save_path=None, fee_rate=base.fee_rate,
                        stop_slippage_pct=base.stop_slippage_pct)
