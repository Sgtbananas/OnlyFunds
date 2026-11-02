"""Audit: independent checks of the backtest engine on REAL data, DEV and VAL periods only.

The final TEST period (2025-01 -> 2026-10) is never touched here.

    python scripts/audit/engine_audit.py --out docs/audit/engine_audit.json

Checks per (coin, strategy, period, params):
  equivalence  current engine == pre-optimisation engine from git commit bc0033a (trades, metrics, decisions)
  timing       every entry fills at the open of the bar after an entry signal (+ slippage)
  accounting   pnl, fees, R-multiple and planned risk recomputed independently from prices
  sizing       notional <= max_position_pct of equity, planned risk <= risk_per_trade of equity
  exits        stop/TP levels honoured; how many TP fills were exact "touches" (optimistic)
  drawdown     max drawdown recomputed independently from the equity curve
  benchmark    buy-and-hold recomputed independently
  gaps         trades held across a missing-candle gap; entries inside the 100-bar gap guard
  halts        runs where the drawdown halt stopped trading early
  boundary     positions still open at the period end (excluded from R statistics)
  history      signals with 500 bars of history == signals with full history inside the window
  lookahead    truncating future data leaves earlier trades unchanged
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core import validation as V  # noqa: E402
from core.backtester import run_backtest  # noqa: E402
from core.core_signals import generate_signals  # noqa: E402
from core.history import DEFAULT_UNIVERSE, load_dataset  # noqa: E402
from core.indicators import warmup_bars  # noqa: E402
from scripts.real_market_validation import base_config  # noqa: E402

PERIODS = {"DEV": V.PERIODS["DEV"], "VAL": V.PERIODS["VAL"]}       # TEST deliberately excluded
H = 3_600_000


def load_reference():
    src = subprocess.run(["git", "show", "bc0033a:(CURRENT) OnlyFunds/core/backtester.py"], cwd=os.path.dirname(ROOT),
                         capture_output=True, text=True, check=True).stdout
    path = os.path.join(tempfile.mkdtemp(), "backtester_ref.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("backtester_ref", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["backtester_ref"] = mod
    spec.loader.exec_module(mod)
    return mod


def same(a, b):
    ta = a.trades.drop(columns="id", errors="ignore")
    tb = b.trades.drop(columns="id", errors="ignore")
    ma = {k: v for k, v in a.metrics.items()}
    mb = {k: v for k, v in b.metrics.items()}
    return ta.equals(tb) and json.dumps(ma, sort_keys=True, default=str) == json.dumps(mb, sort_keys=True, default=str) \
        and a.decisions.equals(b.decisions)


def check_run(seg, cfg, res, sig, start_ts):
    """Independent recomputation for one run. Returns a dict of counters / max errors."""
    o = {"trades": len(res.trades)}
    t = res.trades
    fee, slip, sslip = cfg.fee_rate, cfg.slippage_pct, cfg.stop_slippage_pct
    ts = pd.to_datetime(seg["timestamp"], utc=True)
    idx = {str(x): i for i, x in enumerate(seg["timestamp"].astype(str))}
    errs = {"timing": 0, "pnl": 0.0, "fees": 0.0, "r": 0.0, "notional_over_cap": 0, "risk_over_budget": 0,
            "tp_below_level": 0, "stop_fill_wrong": 0, "exact_tp_touch": 0, "tp_exits": 0,
            "stop_exits_beyond_1r": 0, "gap_crossing_trades": 0}
    cash = cfg.initial_capital
    gaps = np.where(np.diff(seg["created_at"].to_numpy("int64")) > H)[0] + 1   # index of bar after a gap
    for _, tr in t.iterrows():
        ei, xi = idx[tr["entry_time"]], idx[tr["exit_time"]]
        # timing: entry signal on previous bar, fill = open * (1 + slip)
        if not (sig["entry"].iloc[ei - 1] and abs(tr["entry_price"] - seg["Open"].iloc[ei] * (1 + slip)) < 1e-9 * tr["entry_price"]):
            errs["timing"] += 1
        q = tr["qty"]
        exp_entry_fee = q * tr["entry_price"] * fee
        exp_exit_fee = q * tr["exit_price"] * fee
        exp_pnl = q * tr["exit_price"] - exp_exit_fee - (q * tr["entry_price"] + exp_entry_fee)
        errs["pnl"] = max(errs["pnl"], abs(exp_pnl - tr["pnl"]))
        errs["fees"] = max(errs["fees"], abs(exp_entry_fee + exp_exit_fee - tr["fees"]))
        risk = q * (tr["entry_price"] * (1 + fee) - tr["stop_loss"] * (1 - sslip) * (1 - fee))
        errs["r"] = max(errs["r"], abs(tr["pnl"] / risk - tr["r_multiple"]))
        if q * tr["entry_price"] > cfg.risk.max_position_pct * cash * (1 + 1e-9):
            errs["notional_over_cap"] += 1
        if risk > cfg.risk.risk_per_trade * cash * (1 + 1e-9):
            errs["risk_over_budget"] += 1
        hi = seg["High"].iloc[xi]
        if tr["exit_reason"] == "take_profit":
            errs["tp_exits"] += 1
            if hi < tr["take_profit"] - 1e-12 and seg["Open"].iloc[xi] < tr["take_profit"]:
                errs["tp_below_level"] += 1
            if hi <= tr["take_profit"] * 1.0005 and seg["Open"].iloc[xi] < tr["take_profit"]:
                errs["exact_tp_touch"] += 1        # high within 0.05% of the level: a fill is uncertain
        if tr["exit_reason"] == "stop_loss":
            raw = min(tr["stop_loss"], seg["Open"].iloc[xi])
            if abs(tr["exit_price"] - raw * (1 - sslip)) > 1e-9 * raw:
                errs["stop_fill_wrong"] += 1
            if tr["r_multiple"] < -1 - 1e-9:
                errs["stop_exits_beyond_1r"] += 1
        if ((gaps > ei) & (gaps <= xi)).any():
            errs["gap_crossing_trades"] += 1
        cash += tr["pnl"]
    o.update(errs)
    # drawdown, independently
    eq = res.equity
    w = eq[ts >= start_ts]
    curve, lows = w["equity"].to_numpy(), w["equity_low"].to_numpy()
    peaks = np.maximum.accumulate(np.r_[cfg.initial_capital, curve[:-1]])
    o["dd_err"] = abs(float(np.max((peaks - lows) / peaks) * 100) - res.metrics["max_drawdown_pct"])
    # first tradable bar: after the trading window starts AND after the indicator warm-up (as the engine)
    first = max(int((ts < start_ts).sum()), warmup_bars(cfg.indicator_params) + 1)
    bh = (seg["Close"].iloc[-1] * (1 - slip) * (1 - fee) / (seg["Open"].iloc[first] * (1 + slip) * (1 + fee)) - 1) * 100
    o["bh_err"] = abs(bh - res.metrics["buy_and_hold_return_pct"])
    o["halted"] = bool(res.metrics["halted"])
    o["open_at_end"] = res.open_position is not None
    d = res.decisions
    o["gap_blocked_entries"] = int((d["action"] == "DATA_GAP").sum()) if len(d) else 0
    o["cost_blocked_entries"] = int(((d["action"] == "BLOCKED") & d["reason"].str.contains("stop distance")).sum()) if len(d) else 0
    return o


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-equivalence", action="store_true")
    a = ap.parse_args(argv)
    ref = None if a.no_equivalence else load_reference()
    base = base_config()
    frozen = json.load(open(os.path.join(ROOT, "docs", "validation", "frozen_decisions.json"), encoding="utf-8"))
    data = {s: load_dataset(s) for s in DEFAULT_UNIVERSE}
    rows, eq_cases, eq_fail = [], 0, []
    for strat in V.STRATEGIES:
        for pname, params in (("fixed", None), ("tuned_dev_val", frozen["candidates"][strat]["tuned_dev_val"])):
            for per, (start, end) in PERIODS.items():
                for sym, df in data.items():
                    seg = V.segment(df, start, end)
                    cfg = V.config_for(base, strat, params, start)
                    sig = generate_signals(seg, strategy=cfg.strategy, threshold=cfg.threshold,
                                           params=cfg.signal_params, indicator_params=cfg.indicator_params)
                    res = run_backtest(seg, cfg, sym, signals=sig)
                    if ref is not None:
                        rcfg = ref.BacktestConfig(**{k: getattr(cfg, k) for k in cfg.__dataclass_fields__})
                        rres = ref.run_backtest(seg, rcfg, sym, signals=sig)
                        eq_cases += 1
                        if not same(res, rres):
                            eq_fail.append(f"{sym}/{strat}/{pname}/{per}")
                    o = check_run(seg, cfg, res, sig, pd.Timestamp(start, tz="UTC"))
                    rows.append({"symbol": sym, "strategy": strat, "params": pname, "period": per, **o})
                print(f"{strat}/{pname} done ({eq_cases} equivalence cases, {len(eq_fail)} mismatches)", flush=True)
    df_rows = pd.DataFrame(rows)
    # history sufficiency and look-ahead on real data (VAL window, every coin, fixed auto)
    hist_rows = []
    for sym, df in data.items():
        start, end = PERIODS["VAL"]
        short = V.segment(df, start, end, history=500)
        full = V.segment(df, start, end, history=10**7)
        s1 = generate_signals(short)
        s2 = generate_signals(full).iloc[len(full) - len(short):].reset_index(drop=True)
        inwin = pd.to_datetime(short["timestamp"], utc=True) >= pd.Timestamp(start, tz="UTC")
        hist_rows.append({"symbol": sym, "max_signal_diff": float((s1["signal"] - s2["signal"])[inwin].abs().max()),
                          "entry_disagreements": int((s1["entry"] != s2["entry"])[inwin].sum()),
                          "regime_disagreements": int((s1["regime"].astype(str) != s2["regime"].astype(str))[inwin].sum())})
        cfg = V.config_for(base, "auto", None, start)
        full_res = run_backtest(short, cfg, sym)
        cut = len(short) - 3000
        trunc_res = run_backtest(short.iloc[:cut].reset_index(drop=True), cfg, sym)
        cut_ts = short["timestamp"].iloc[cut - 1]
        a_ = full_res.trades[pd.to_datetime(full_res.trades["exit_time"], utc=True) < cut_ts] if len(full_res.trades) else full_res.trades
        b_ = trunc_res.trades[pd.to_datetime(trunc_res.trades["exit_time"], utc=True) < cut_ts] if len(trunc_res.trades) else trunc_res.trades
        hist_rows[-1]["lookahead_trades_compared"] = int(len(a_))
        hist_rows[-1]["lookahead_identical"] = bool(a_.drop(columns="id", errors="ignore").reset_index(drop=True).equals(
            b_.drop(columns="id", errors="ignore").reset_index(drop=True)))
    summary = {
        "runs": int(len(df_rows)), "trades_checked": int(df_rows["trades"].sum()),
        "equivalence_cases": eq_cases, "equivalence_mismatches": eq_fail,
        "timing_errors": int(df_rows["timing"].sum()), "max_pnl_error": float(df_rows["pnl"].max()),
        "max_fee_error": float(df_rows["fees"].max()), "max_r_error": float(df_rows["r"].max()),
        "notional_over_cap": int(df_rows["notional_over_cap"].sum()),
        "risk_over_budget": int(df_rows["risk_over_budget"].sum()),
        "tp_fill_without_touch": int(df_rows["tp_below_level"].sum()),
        "stop_fill_wrong": int(df_rows["stop_fill_wrong"].sum()),
        "tp_exits": int(df_rows["tp_exits"].sum()), "tp_exact_touch_fills": int(df_rows["exact_tp_touch"].sum()),
        "stop_exits_beyond_1r": int(df_rows["stop_exits_beyond_1r"].sum()),
        "gap_crossing_trades": int(df_rows["gap_crossing_trades"].sum()),
        "max_drawdown_error_pct": float(df_rows["dd_err"].max()), "max_buy_hold_error_pct": float(df_rows["bh_err"].max()),
        "runs_halted_by_drawdown": int(df_rows["halted"].sum()),
        "halted_runs": df_rows[df_rows["halted"]][["symbol", "strategy", "params", "period"]].values.tolist(),
        "runs_with_position_open_at_end": int(df_rows["open_at_end"].sum()),
        "gap_blocked_entries": int(df_rows["gap_blocked_entries"].sum()),
        "cost_blocked_entries": int(df_rows["cost_blocked_entries"].sum()),
        "history_check": hist_rows,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "runs": rows}, f, indent=1, default=str)
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
