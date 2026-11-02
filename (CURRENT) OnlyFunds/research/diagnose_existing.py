"""Milestone 1: diagnose WHY the four existing strategies lose money.

Re-runs the unchanged engine (fixed parameters, Conservative risk, CoinEx-level costs) on DEV+VAL only
(2019-01-01 -> 2025-01-01) at 1h, 4h and 1d. The consumed 2025-26 TEST window is never loaded.
Every trade is enriched with conditions at entry and its cost is decomposed.

    python research/diagnose_existing.py          -> docs/research/01_diagnosis.md
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from core import validation as V  # noqa: E402
from core.history import DEFAULT_UNIVERSE, load_dataset  # noqa: E402
from utils.config import build_settings, load_config  # noqa: E402

START, END = "2019-01-01", "2025-01-01"          # DEV + VAL; TEST excluded
INTERVALS = ("1h", "4h", "1d")
OUT = os.path.join(ROOT, "docs", "research", "01_diagnosis.md")
STOPS = {"stop_loss", "trailing_stop"}


def enrich(df, trades, cfg, interval):
    """Attach entry conditions and gross/cost decomposition to engine trades."""
    bars_per_day = {"1h": 24, "4h": 6, "1d": 1}[interval]
    c = df["Close"]
    ret = np.log(c).diff()
    vol = ret.rolling(30 * bars_per_day, min_periods=10).std()
    trend = c / c.ewm(span=200).mean() - 1          # price vs 200-bar EMA
    liq = (df["QuoteVolume"]).rolling(bars_per_day * 7, min_periods=1).median()
    feat = pd.DataFrame({"timestamp": df["timestamp"], "vol": vol.shift(1), "trend": trend.shift(1),
                         "liq": liq.shift(1)})
    t = trades.copy()
    t["entry_ts"] = pd.to_datetime(t["entry_time"], utc=True)
    t = t.merge(feat, left_on="entry_ts", right_on="timestamp", how="left")
    raw_entry = t["entry_price"] / (1 + cfg.slippage_pct)
    exit_slip = np.where(t["exit_reason"].isin(STOPS), cfg.stop_slippage_pct, cfg.slippage_pct)
    raw_exit = t["exit_price"] / (1 - exit_slip)
    t["gross_pnl"] = t["qty"] * (raw_exit - raw_entry)
    t["gross_r"] = t["gross_pnl"] / t["risk_amount"]
    t["cost_r"] = t["gross_r"] - t["r_multiple"]
    t["fee_r"] = t["fees"] / t["risk_amount"]
    t["stop_dist_pct"] = (t["entry_price"] - t["stop_loss"]) / t["entry_price"] * 100
    t["hour"] = t["entry_ts"].dt.hour
    t["weekday"] = t["entry_ts"].dt.day_name().str[:3]
    t["year"] = t["entry_ts"].dt.year
    return t


def run_all():
    base_cfg = load_config()
    rows = []
    for interval in INTERVALS:
        data = {s: load_dataset(s, interval) for s in DEFAULT_UNIVERSE}
        bt, _ = build_settings(base_cfg, "Conservative", {"interval": interval})
        bt.initial_capital = 1000.0
        for strat in V.STRATEGIES:
            res = V.run_period(data, bt, strat, None, START, END)
            for pair, r in res.items():
                if len(r.trades):
                    t = enrich(V.segment(data[pair], START, END), r.trades, bt, interval)
                    rows.append(t.assign(interval=interval, strategy_cfg=strat, pair=pair))
            print(interval, strat, sum(len(r.trades) for r in res.values()), flush=True)
    return pd.concat(rows, ignore_index=True), bt


def agg(t, by):
    g = t.groupby(by, observed=True)
    out = pd.DataFrame({"trades": g.size(), "net_R": g["r_multiple"].mean(), "gross_R": g["gross_r"].mean(),
                        "cost_R": g["cost_r"].mean(), "win%": g["r_multiple"].apply(lambda r: (r > 0).mean() * 100),
                        "total_net_R": g["r_multiple"].sum()})
    return out.reset_index()


def md(df):
    d = df.copy()
    for col in d.columns:
        if d[col].dtype.kind == "f":
            d[col] = d[col].map(lambda v: f"{v:+.3f}" if abs(v) < 100 else f"{v:.0f}")
    return ("| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns) + "\n" +
            "\n".join("| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False)))


def main():
    cache = os.path.join(ROOT, "state", "research", "diagnosis_trades.csv")
    if "--reuse" in sys.argv and os.path.exists(cache):          # report only, from the saved engine trades
        t = pd.read_csv(cache)
        t["entry_ts"] = pd.to_datetime(t["entry_ts"], utc=True)
        bt, _ = build_settings(load_config(), "Conservative", {"interval": "1h"})
    else:
        t, bt = run_all()
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        t.to_csv(cache, index=False)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for col, q in (("vol", "vol_tercile"), ("trend", "trend_tercile"), ("liq", "liq_tercile")):
        t[q] = t.groupby(["interval", "pair"])[col].transform(
            lambda s: pd.qcut(s.rank(method="first"), 3, labels=["low", "mid", "high"]) if s.notna().sum() >= 3 else np.nan)
    t["hold_bucket"] = pd.cut(t["bars_held"], [-1, 2, 6, 24, 72, 10**6], labels=["0-2", "3-6", "7-24", "25-72", ">72"])
    t["r_bucket"] = pd.cut(t["r_multiple"], [-10, -1.05, -0.5, 0, 0.5, 1, 2, 50],
                           labels=["<-1.05", "-1.05..-0.5", "-0.5..0", "0..0.5", "0.5..1", "1..2", ">2"])
    sections = [
        ("Interval x strategy (the headline)", ["interval", "strategy_cfg"]),
        ("By coin (all intervals and strategies)", ["interval", "pair"]),
        ("By regime at entry", ["interval", "regime"]),
        ("By entry signal (strategy that fired)", ["interval", "strategy"]),
        ("By exit reason", ["interval", "exit_reason"]),
        ("By holding duration (bars)", ["interval", "hold_bucket"]),
        ("Volatility at entry (tercile within coin)", ["interval", "vol_tercile"]),
        ("Trend strength at entry (price vs slow EMA, tercile)", ["interval", "trend_tercile"]),
        ("Liquidity at entry (7-day median quote volume, tercile)", ["interval", "liq_tercile"]),
        ("Year", ["interval", "year"]),
        ("Day of week (1h only)", ["weekday"]),
        ("Hour of day UTC (1h only, 4-hour blocks)", ["hour_block"]),
    ]
    t["hour_block"] = (t["hour"] // 4 * 4).astype(str).str.zfill(2) + "h"
    out = ["# Milestone 1: why the existing strategies lose money\n",
           f"Engine unchanged; fixed parameters; Conservative risk; fees {bt.fee_rate:.2%}/side, slippage "
           f"{bt.slippage_pct:.2%}, stop slippage {bt.stop_slippage_pct:.2%}; long-only spot; 9 Binance coins; "
           f"**DEV+VAL only ({START} -> {END}); the consumed 2025-26 TEST window is not loaded.**\n",
           "R = planned loss at the stop including costs. gross_R = the same trade with zero fees and zero slippage; "
           "cost_R = gross_R - net_R.\n"]
    dist = t.groupby("interval")["r_bucket"].value_counts(normalize=True).unstack().mul(100).round(1)
    cost = t.groupby("interval").agg(trades=("r_multiple", "size"), median_stop_dist_pct=("stop_dist_pct", "median"),
                                     mean_cost_R=("cost_r", "mean"), mean_fee_R=("fee_r", "mean"),
                                     mean_gross_R=("gross_r", "mean"), mean_net_R=("r_multiple", "mean")).reset_index()
    out.append("## Costs versus stop distance\n\n" + md(cost) + "\n")
    out.append("## Win/loss distribution (% of trades by net R)\n\n" + md(dist.reset_index()) + "\n")
    for title, by in sections:
        sub = t[t["interval"] == "1h"] if "1h only" in title else t
        out.append(f"## {title}\n\n" + md(agg(sub, by)) + "\n")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(open(OUT, encoding="utf-8").read()[:6000])


if __name__ == "__main__":
    main()
