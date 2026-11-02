"""Prospective (forward) shadow tracking of the frozen research candidates T50 and RS3.

RESEARCH ONLY. No exchange orders, no keys, no capital. Reads public Binance daily klines only.
Neither strategy is approved for trading; LIVE_TRADING_LOCKED is untouched.

    python research/forward_shadow.py freeze     # once: writes research/shadow/FROZEN.json (refuses to overwrite)
    python research/forward_shadow.py update     # daily: snapshot data, recompute both books, write run + report

Design
- Frozen spec + fingerprint: rules, parameters, universes, costs, risk limits, benchmark definitions and the
  SHA-256 of every implementing file (line endings normalised). `update` refuses to run if anything changed.
- Two books per strategy, kept completely separate between strategies:
  1. signal book: the unchanged per-coin engine (one position per coin, 0.5% risk sizing). Every entry signal
     and its fate (filled / blocked / skipped) is recorded; this is the evidence sample.
  2. portfolio book: one 10,000 USDT account with the Conservative limits across coins (research/portfolio.py),
     recording every accepted and rejected signal with the reason.
- Deterministic recomputation from the start date on each run, from data snapshotted at that run. Each run is
  stored under state/shadow/runs/<run>/; records dated before the previous run's last complete day must be
  identical to that run's, otherwise a DATA_REVISION anomaly is logged (nothing is overwritten).
- Indicator warm-up uses candles before the start date only to compute indicators; no trade, signal or
  statistic before the start date is ever evaluated (that history lies in the consumed 2025-26 window).
"""
import argparse
import copy
import glob
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))

FROZEN = os.path.join(ROOT, "research", "shadow", "FROZEN.json")
STATE = os.path.join(os.environ.get("OF2_STATE_DIR") or os.path.join(ROOT, "state"), "shadow")
START = "2026-11-02"            # first possible fill (decision at the close of the 2026-11-01 daily candle)
WARMUP_START = "2026-05-01"     # indicator history only (>= 100-day BTC average + ATR settling); never evaluated
MAIN = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "LTCUSDT", "ADAUSDT", "EOSUSDT", "XLMUSDT", "TRXUSDT"]
HOLDOUT = ["ETCUSDT", "LINKUSDT", "NEOUSDT", "QTUMUSDT", "IOTAUSDT", "ONTUSDT", "ICXUSDT", "WAVESUSDT", "BATUSDT",
           "VETUSDT", "ZILUSDT", "NULSUSDT", "ZECUSDT", "DASHUSDT", "XMRUSDT", "OMGUSDT"]
IMPLEMENTATION = ["research/forward_shadow.py", "research/round1.py", "research/portfolio.py", "core/backtester.py",
                  "core/core_signals.py", "core/indicators.py", "core/positions.py", "core/risk_manager.py",
                  "core/validation.py", "core/regime.py", "core/history.py", "utils/config.py", "config/config.yaml"]
N_RANDOM, SEED = 1000, 2611
CLUSTER_BOOT = 10_000


# ------------------------------------------------------------------ freezing
def file_hash(rel):
    with open(os.path.join(ROOT, rel), "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def spec():
    return {
        "purpose": "prospective validation only; not approved for trading; no orders; live trading stays locked",
        "start_first_fill_utc": START, "warmup_history_from": WARMUP_START,
        "data_source": "Binance spot public REST /api/v3/klines, interval 1d, completed candles only, UTC days",
        "execution": "decision at the daily close; fill at the next daily open with slippage; stops checked "
                     "intrabar (gap through the stop fills at the open); stop fills get stop slippage; fees both sides",
        "costs": {"fee_per_side": 0.002, "slippage": 0.0005, "stop_slippage": 0.002, "volume_cap_of_prev_bar": 0.10,
                  "max_cost_to_risk": 0.25},
        "risk_conservative": {"risk_per_trade": 0.005, "max_position_pct": 0.10, "max_exposure_pct": 0.30,
                              "max_open_positions": 3, "max_daily_loss_pct": 0.03, "max_drawdown_pct": 0.10,
                              "drawdown_halt": "permanent (kill switch)", "portfolio_capital_usdt": 10000,
                              "signal_book_capital_per_coin_usdt": 1000},
        "strategies": {
            "T50": {
                "universe": MAIN + HOLDOUT,
                "universe_rule": "main 9 OF2 coins + the 16-coin holdout fixed by rule in RESEARCH_PROGRAMME.md; "
                                 "a coin with no live Binance data (e.g. delisted EOS) is recorded as unavailable",
                "entry": "daily close > highest close of the previous 50 days",
                "exit": "daily close < lowest close of the previous 25 days -> sell next open",
                "stop": "3 x ATR(14) below the fill price, fixed; no take-profit (tp_atr_mult=1000); no trailing",
                "signal_code": "research/round1.py: signals('T50')",
                "portfolio_order": "universe order when slots are short",
            },
            "RS3": {
                "universe": MAIN,
                "entry": "at Monday's daily close: coin in the top 3 of the universe by 28-day return, its 28-day "
                         "return > 0, and BTC close > BTC 100-day simple average",
                "exit": "at Monday's close if any entry condition fails -> sell next open",
                "stop": "3 x ATR(14) below the fill price, fixed; no take-profit; no trailing",
                "signal_code": "research/round1.py: rs_basket(), signals('RS3')",
                "portfolio_order": "universe order when slots are short",
            },
        },
        "random_benchmark": {
            "T50": "per coin, entry on each day with probability p_coin = share of 2019-2024 days on which T50's "
                   "entry condition was true for that coin (frozen values below); same exits, stop, sizing, costs",
            "RS3": "on each Monday on which the BTC filter is on, 3 coins drawn at random from those with data; "
                   "same exits (leave at the next Monday), stop, sizing, costs",
            "runs": N_RANDOM, "seed": SEED,
        },
        "confidence_intervals": "cluster bootstrap resampling calendar weeks of entry (10,000 resamples); results "
                                "also reported by BTC market state (core/regime.detect on BTC daily)",
        "pass_criteria_prospective_sample_only": [
            ">= 100 signal-book trades in the prospective sample (no historical trades counted)",
            "expectancy > 0R and the week-clustered 95% bootstrap lower bound > 0",
            "profit factor >= 1.10",
            "positive on >= 60% of coins with >= 10 trades; still positive without the best coin",
            "expectancy above the 95th percentile of the frozen random-entry benchmark over the same period",
        ],
        "reporting": "every update writes a run record; interim results never change the rules",
        "implementation_sha256": {f: file_hash(f) for f in IMPLEMENTATION},
    }


def canonical(d):
    return json.dumps(d, sort_keys=True, separators=(",", ":"), default=str).encode()


def fingerprint(s, name):
    sub = {k: v for k, v in s.items() if k not in ("strategies", "random_benchmark", "fingerprints", "frozen_at_utc",
                                                   "random_entry_probabilities", "git_commit")}
    return hashlib.sha256(canonical({**sub, "strategy": s["strategies"][name],
                                     "benchmark": s["random_benchmark"][name],
                                     "p": s.get("random_entry_probabilities", {}) if name == "T50" else {}})).hexdigest()


def historical_entry_rates():
    """Frozen per-coin T50 entry-signal frequency from 2019-2024 (no data >= 2025 is loaded)."""
    import round1 as R1
    data = R1.load(MAIN) | R1.load(HOLDOUT)
    out = {}
    for pair, df in data.items():
        c = df["Close"]
        out[pair] = float((c > c.rolling(50).max().shift(1)).mean())
    return out


def cmd_freeze():
    if os.path.exists(FROZEN):
        print(f"REFUSED: {FROZEN} exists; the strategies are already frozen.")
        return 2
    s = spec()
    s["random_entry_probabilities"] = historical_entry_rates()
    s["frozen_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        import subprocess
        s["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                         text=True).stdout.strip()
    except Exception:
        s["git_commit"] = "unknown"
    s["fingerprints"] = {n: fingerprint(s, n) for n in s["strategies"]}
    os.makedirs(os.path.dirname(FROZEN), exist_ok=True)
    with open(FROZEN, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=1, sort_keys=True)
    print(json.dumps(s["fingerprints"], indent=1))
    return 0


def load_frozen():
    with open(FROZEN, encoding="utf-8") as f:
        fz = json.load(f)
    problems = []
    for rel, h in fz["implementation_sha256"].items():
        if not os.path.exists(os.path.join(ROOT, rel)):
            problems.append(f"{rel} missing")
        elif file_hash(rel) != h:
            problems.append(f"{rel} changed since freezing")
    for n, fp in fz["fingerprints"].items():
        if fingerprint(fz, n) != fp:
            problems.append(f"{n} spec fingerprint mismatch")
    return fz, problems


# ------------------------------------------------------------------ data
def _previous_snapshot(symbol, run_dir):
    """Most recent earlier snapshot of ``symbol`` (keeps a coin's record after Binance stops returning it)."""
    for d in sorted(glob.glob(os.path.join(STATE, "runs", "*")), reverse=True):
        p = os.path.join(d, "data", f"{symbol}_1d.csv")
        if d != run_dir and os.path.exists(p):
            return p
    return None


def snapshot(run_dir, symbols):
    """Download completed daily candles. Returns (data, status, fatal).

    - A coin Binance no longer returns (empty response or HTTP 400 invalid symbol) keeps its record: its most
      recent earlier snapshot is reused and marked; if it never had one, it is recorded as unavailable.
    - A coin whose data stopped updating ("stale") stays in the evaluation up to its last candle, so trades and
      open positions are never dropped; it no longer limits the evaluation date.
    - Any other download failure makes the whole run fatal: nothing is computed from a partial universe.
    """
    from core.history import fetch_binance
    end = (pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    data, status, fatal = {}, {}, []
    os.makedirs(os.path.join(run_dir, "data"), exist_ok=True)
    for s in symbols:
        df, note = None, None
        try:
            df = fetch_binance(s, "1d", WARMUP_START, end)
        except Exception as exc:
            msg = str(exc)
            if "HTTP 400" in msg:
                df = pd.DataFrame()
                note = "Binance HTTP 400 (symbol not trading)"
            else:
                fatal.append(f"{s}: {msg[:160]}")
                continue
        if df is None or df.empty:
            prev = _previous_snapshot(s, run_dir)
            if prev is None:
                status[s] = f"no data returned (delisted or not trading){'; ' + note if note else ''}"
                continue
            df = pd.read_csv(prev)
            note = f"no longer returned by Binance; using last snapshot {os.path.relpath(prev, STATE)}"
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        path = os.path.join(run_dir, "data", f"{s}_1d.csv")
        df.to_csv(path, index=False)
        with open(path, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        days = df["timestamp"]
        missing = int(((days.diff().dt.days.fillna(1)) - 1).clip(lower=0).sum())
        last = days.max()
        status[s] = {"rows": len(df), "first": f"{days.min():%Y-%m-%d}", "last_complete_day": f"{last:%Y-%m-%d}",
                     "missing_days": missing, "sha256": h,
                     "stale": bool(last < pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=2)),
                     **({"note": note} if note else {})}
        data[s] = df
    return data, status, fatal


# ------------------------------------------------------------------ books
def signal_book(name, data, ctx, cfg, start, end, rng=None, probs=None, rand_member=None):
    """Per-coin engine; returns trades, open positions and every entry decision."""
    import round1 as R1
    from core import validation as V
    from core.backtester import run_backtest
    trades, decisions, open_pos = [], [], []
    for pair, df in data.items():
        seg = V.segment(df, WARMUP_START, end, history=0)
        ts = pd.to_datetime(seg["timestamp"], utc=True)
        if (ts >= pd.Timestamp(start, tz="UTC")).sum() < 1:
            continue
        c = copy.deepcopy(cfg)
        c.strategy, c.trade_from = "trend_following", start
        if rng is None:
            sig = R1.signals(name, seg, {**ctx, "pair": pair})
        elif name == "T50":
            sig = R1.signals("RAND_T50", seg, ctx, rng=rng, p=probs[pair])
        else:
            sig = R1.signals("RAND_RS", seg, {**ctx, "pair": pair, "rand_member": rand_member})
        r = run_backtest(seg, c, pair, signals=sig)
        if len(r.trades):
            trades.append(r.trades.assign(pair=pair))
        if len(r.decisions):
            decisions.append(r.decisions.assign(pair=pair))
        if r.open_position:
            open_pos.append({**r.open_position, "pair": pair, "mark": float(seg["Close"].iloc[-1])})
    t = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    d = pd.concat(decisions, ignore_index=True) if decisions else pd.DataFrame()
    return t, d, pd.DataFrame(open_pos)


def portfolio_book(name, data, ctx, fz, start, end):
    import round1 as R1
    from core.risk_manager import RiskLimits
    from portfolio import Costs, simulate, stats
    frames, ent, ex, atr = {}, {}, {}, {}
    for pair, df in data.items():
        sig = R1.signals(name, df, {**ctx, "pair": pair})
        idx = pd.to_datetime(df["timestamp"], utc=True)
        frames[pair] = df.set_index(idx)[["Open", "High", "Low", "Close", "Volume"]]
        ent[pair] = pd.Series(sig["entry"].to_numpy(bool), index=idx)
        ex[pair] = pd.Series(sig["exit"].to_numpy(bool), index=idx)
        atr[pair] = pd.Series(sig["ATR"].to_numpy(float), index=idx)
    rk = fz["risk_conservative"]
    lim = RiskLimits(risk_per_trade=rk["risk_per_trade"], max_position_pct=rk["max_position_pct"],
                     max_exposure_pct=rk["max_exposure_pct"], max_open_positions=rk["max_open_positions"],
                     max_daily_loss_pct=rk["max_daily_loss_pct"], max_drawdown_pct=rk["max_drawdown_pct"],
                     min_notional=1.0, max_cost_to_risk=fz["costs"]["max_cost_to_risk"], kill_switch_file="")
    co = fz["costs"]
    log = []
    # the last day's decisions need tomorrow's open: simulate up to 'end' (exclusive) without force-closing
    res = simulate(frames, ent, ex, atr, start, end, lim, Costs(co["fee_per_side"], co["slippage"], co["stop_slippage"]),
                   capital=rk["portfolio_capital_usdt"], close_at_end=False, log=log)
    return res, pd.DataFrame(log), stats(res, rk["portfolio_capital_usdt"]) if len(res.equity) else {}


# ------------------------------------------------------------------ statistics
def week_cluster_ci(t, n=CLUSTER_BOOT, seed=5):
    if len(t) < 2:
        return float("nan"), float("nan")
    wk = pd.to_datetime(t["entry_time"], utc=True).dt.strftime("%G-%V")
    groups = [g["r_multiple"].to_numpy(float) for _, g in t.groupby(wk)]
    rng = np.random.default_rng(seed)
    k = len(groups)
    sums = np.array([g.sum() for g in groups])
    cnts = np.array([len(g) for g in groups])
    idx = rng.integers(0, k, size=(n, k))
    means = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def btc_market(btc):
    from core.regime import detect
    lab = detect(btc)["trend"].map({"strong_up": "bull", "weak_up": "bull", "range": "sideways",
                                    "weak_down": "bear", "strong_down": "bear"})
    return pd.Series(lab.to_numpy(), index=pd.to_datetime(btc["timestamp"], utc=True))


def breakdown(t, col):
    return {str(k): {"trades": int(len(g)), "net_R": float(g.r_multiple.mean()), "total_R": float(g.r_multiple.sum())}
            for k, g in t.groupby(col)}


def summarise(t, open_pos, btc_lab=None):
    if t is None or len(t) == 0:
        return {"closed_trades": 0, "open_positions": int(len(open_pos)),
                "open_detail": open_pos.to_dict("records") if len(open_pos) else []}
    t = t.copy()
    if btc_lab is not None:
        dec = pd.to_datetime(t["entry_time"], utc=True) - pd.Timedelta(days=1)
        t["btc_market"] = btc_lab.reindex(dec).fillna("n/a").to_numpy()
    gl = -t.loc[t.pnl <= 0, "pnl"].sum()
    per = t.groupby("pair")["r_multiple"].agg(["count", "mean"])
    elig = per[per["count"] >= 10]
    best = t.groupby("pair")["r_multiple"].sum().idxmax()
    lo, hi = week_cluster_ci(t)
    return {"closed_trades": int(len(t)), "open_positions": int(len(open_pos)), "net_R": float(t.r_multiple.mean()),
            "median_R": float(t.r_multiple.median()), "ci95_week_cluster": [lo, hi],
            "profit_factor": float(t.loc[t.pnl > 0, "pnl"].sum() / gl) if gl > 0 else None,
            "win_rate_pct": float((t.pnl > 0).mean() * 100), "fees_usdt": float(t["fees"].sum()),
            "net_pnl_usdt": float(t.pnl.sum()), "coins_traded": int(per.shape[0]),
            "breadth_coins_ge10": float((elig["mean"] > 0).mean()) if len(elig) else None,
            "without_best_coin_R": float(t[t.pair != best].r_multiple.mean()) if (t.pair != best).any() else None,
            "by_coin": breakdown(t, "pair"), "by_btc_market_at_decision": breakdown(t, "btc_market")
            if "btc_market" in t else {}, "open_detail": open_pos.to_dict("records") if len(open_pos) else []}


def random_distribution(name, data, ctx, cfg, fz, start, end, member=None, btc_ok=None):
    rng = np.random.default_rng(fz["random_benchmark"]["seed"])
    probs = fz.get("random_entry_probabilities", {})
    vals = []
    for _ in range(fz["random_benchmark"]["runs"]):
        rm = None
        if name == "RS3":
            idx = member.index[(member.index >= pd.Timestamp(start, tz="UTC") - pd.Timedelta(days=1))]
            rm = pd.DataFrame(False, index=member.index, columns=member.columns)
            on = btc_ok.reindex(member.index).fillna(False).astype(bool)
            for d in idx[(idx.dayofweek == 0) & on.reindex(idx).to_numpy()]:
                avail = [c for c in member.columns if pd.notna(ctx["closes"].loc[d, c])]
                if len(avail) >= 3:
                    rm.loc[d, rng.choice(avail, 3, replace=False)] = True
        t, _, _ = signal_book(name, data, ctx, cfg, start, end, rng=rng, probs=probs, rand_member=rm)
        vals.append(float(t.r_multiple.mean()) if len(t) else np.nan)
    v = np.array(vals, float)
    return v


# ------------------------------------------------------------------ update
def cmd_update(skip_random=False):
    import round1 as R1
    fz, problems = load_frozen()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(STATE, "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    rec = {"run_id": run_id, "frozen_fingerprints": fz["fingerprints"], "problems": problems}
    if problems:
        rec["status"] = "STOPPED: frozen implementation or spec changed; tracking refused (results preserved)"
        _write(run_dir, "run.json", rec)
        print(rec["status"], problems)
        return 3
    symbols = sorted(set(fz["strategies"]["T50"]["universe"]) | set(fz["strategies"]["RS3"]["universe"]))
    data, status, fatal = snapshot(run_dir, symbols)
    rec["data_status"] = status
    if fatal:
        rec["status"] = "FETCH_FAILED: nothing computed (no partial-universe evaluation)"
        rec["fetch_errors"] = fatal
        _write(run_dir, "run.json", rec)
        print(rec["status"], fatal)
        return 5
    if "BTCUSDT" not in data:
        rec["status"] = "NO BTC DATA: cannot evaluate filters; nothing computed"
        _write(run_dir, "run.json", rec)
        return 4
    fresh = [s for s, v in status.items() if isinstance(v, dict) and not v["stale"]]
    if not fresh:
        rec["status"] = "NO FRESH DATA: nothing computed"
        _write(run_dir, "run.json", rec)
        return 4
    last_day = min(pd.to_datetime(data[s]["timestamp"], utc=True).max() for s in fresh)
    end = (last_day + pd.Timedelta(days=1)).strftime("%Y-%m-%d")     # exclusive
    rec["evaluated_through_utc_day"] = f"{last_day:%Y-%m-%d}"
    if pd.Timestamp(end, tz="UTC") <= pd.Timestamp(START, tz="UTC"):
        rec["status"] = f"waiting: no completed day on/after the start ({START}) yet"
        _write(run_dir, "run.json", rec)
        print(rec["status"])
        return 0
    btc_ok = R1.btc_filter(data["BTCUSDT"])
    btc_lab = btc_market(data["BTCUSDT"])
    cfg = R1.config()
    cfg.close_at_end = False
    out = {}
    for name in ("T50", "RS3"):
        uni = {s: data[s] for s in fz["strategies"][name]["universe"] if s in data}
        member, _ = R1.rs_basket(uni, btc_ok)
        closes = pd.DataFrame({s: d.set_index(pd.to_datetime(d["timestamp"], utc=True))["Close"] for s, d in uni.items()})
        ctx = {"btc_ok": btc_ok, "member": member, "closes": closes}
        t, dec, op = signal_book(name, uni, ctx, cfg, START, end)
        pres, plog, pstats = portfolio_book(name, uni, ctx, fz, START, end)
        sdir = os.path.join(run_dir, name)
        os.makedirs(sdir, exist_ok=True)
        for fname, df in (("signal_book_trades.csv", t), ("signal_book_decisions.csv", dec),
                          ("signal_book_open.csv", op), ("portfolio_trades.csv", pres.trades),
                          ("portfolio_signal_log.csv", plog),
                          ("portfolio_equity.csv", pd.DataFrame({"equity": pres.equity, "exposure": pres.exposure,
                                                                 "open_positions": pres.n_open}))):
            df.to_csv(os.path.join(sdir, fname), index=fname == "portfolio_equity.csv")
        summ = {"signal_book": summarise(t, op, btc_lab), "portfolio_book": pstats,
                "signals_recorded": {"signal_book_decisions": int(len(dec)),
                                     "portfolio_signal_log": plog["status"].value_counts().to_dict() if len(plog) else {}},
                "universe_available": sorted(uni), "universe_unavailable": sorted(set(fz["strategies"][name]["universe"]) - set(uni))}
        if not skip_random and len(t):
            v = random_distribution(name, uni, ctx, cfg, fz, START, end, member=member, btc_ok=btc_ok)
            v = v[np.isfinite(v)]
            summ["random_benchmark"] = {"runs_with_trades": int(len(v)), "mean_R": float(v.mean()) if len(v) else None,
                                        "p95_R": float(np.percentile(v, 95)) if len(v) else None,
                                        "strategy_percentile": float((v < summ["signal_book"]["net_R"]).mean() * 100)
                                        if len(v) else None}
        out[name] = summ
    rec["results"] = out
    rec["anomalies"] = check_revisions(run_dir)
    rec["status"] = "ok"
    _write(run_dir, "run.json", rec)
    write_report(rec, fz)
    print(json.dumps({k: rec[k] for k in ("run_id", "status", "evaluated_through_utc_day", "anomalies")}, indent=1))
    return 0


def check_revisions(run_dir):
    """Records dated before the previous run's last evaluated day must not change (data revisions / bugs)."""
    runs = sorted(d for d in glob.glob(os.path.join(STATE, "runs", "*")) if d != run_dir)
    prev = None
    for d in reversed(runs):
        try:
            with open(os.path.join(d, "run.json"), encoding="utf-8") as f:
                j = json.load(f)
            if j.get("status") == "ok":
                prev = (d, j)
                break
        except Exception:
            continue
    if not prev:
        return []
    pdir, pj = prev
    cutoff = pd.Timestamp(pj["evaluated_through_utc_day"], tz="UTC")
    issues = []
    for name in ("T50", "RS3"):
        a = _read(os.path.join(pdir, name, "signal_book_trades.csv"))
        b = _read(os.path.join(run_dir, name, "signal_book_trades.csv"))
        if a is None or b is None:
            continue
        a = a[pd.to_datetime(a["exit_time"], utc=True) < cutoff].reset_index(drop=True)
        b = b[pd.to_datetime(b["exit_time"], utc=True) < cutoff].reset_index(drop=True)
        cols = ["pair", "entry_time", "exit_time", "entry_price", "exit_price", "r_multiple"]
        if len(a) != len(b) or not np.allclose(a[cols[3:]].to_numpy(float), b[cols[3:]].to_numpy(float), rtol=1e-9) \
                or not (a[cols[:3]].astype(str).to_numpy() == b[cols[:3]].astype(str).to_numpy()).all():
            issues.append(f"DATA_REVISION: {name} closed trades before {cutoff:%Y-%m-%d} differ from run {os.path.basename(pdir)}")
    return issues


def _read(p):
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def _write(d, name, obj):
    with open(os.path.join(d, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str)


# ------------------------------------------------------------------ report
def write_report(rec, fz):
    lines = ["# OF2 forward shadow tracking: T50 and RS3 (research only)\n",
             "**Not approved for trading. No orders, no capital, no keys. Live trading stays locked.** "
             "Prospective results are reported separately from all historical results and are never combined.\n",
             f"- Run: `{rec['run_id']}`; evaluated through UTC day **{rec.get('evaluated_through_utc_day')}**",
             f"- Start (first possible fill): **{START}**; frozen at {fz['frozen_at_utc']} (commit `{fz.get('git_commit', '?')[:12]}`)",
             f"- Data: {fz['data_source']}",
             f"- Execution: {fz['execution']}",
             f"- Costs: fee {fz['costs']['fee_per_side']:.2%}/side, slippage {fz['costs']['slippage']:.2%}, stop slippage "
             f"{fz['costs']['stop_slippage']:.2%}; Conservative limits (portfolio book)\n",
             "## Frozen fingerprints\n", "| strategy | SHA-256 |", "|---|---|"]
    lines += [f"| {n} | `{fp}` |" for n, fp in fz["fingerprints"].items()]
    for name, r in rec.get("results", {}).items():
        sb, pb = r["signal_book"], r["portfolio_book"]
        lines += [f"\n## {name}\n", f"Rules: entry: {fz['strategies'][name]['entry']}; exit: {fz['strategies'][name]['exit']}; "
                  f"stop: {fz['strategies'][name]['stop']}.\n",
                  f"Universe available: {len(r['universe_available'])} coins; unavailable: {', '.join(r['universe_unavailable']) or 'none'}.\n",
                  "Signal book (evidence sample, per-coin engine):\n", "```json", json.dumps(sb, indent=1, default=str), "```",
                  "Portfolio book (one 10,000 USDT account, Conservative limits):\n", "```json",
                  json.dumps(pb, indent=1, default=str), "```"]
        if "random_benchmark" in r:
            lines += ["Random-entry benchmark (frozen definition, same period):\n", "```json",
                      json.dumps(r["random_benchmark"], indent=1, default=str), "```"]
        n = sb.get("closed_trades", 0)
        lines += [f"Outstanding validation requirement: {max(0, 100 - n)} more closed signal-book trades before the "
                  "pre-registered criteria can even be evaluated; all criteria are listed in FROZEN.json.\n"]
    lines += ["## Data status and anomalies\n", "```json", json.dumps(rec.get("data_status", {}), indent=1, default=str), "```",
              f"Anomalies: {rec.get('anomalies') or 'none'}\n"]
    os.makedirs(os.path.join(STATE, "reports"), exist_ok=True)
    txt = "\n".join(lines)
    with open(os.path.join(STATE, "reports", f"{rec['run_id']}.md"), "w", encoding="utf-8") as f:
        f.write(txt)
    with open(os.path.join(STATE, "SHADOW_REPORT_LATEST.md"), "w", encoding="utf-8") as f:
        f.write(txt)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["freeze", "update"])
    ap.add_argument("--skip-random", action="store_true")
    a = ap.parse_args(argv)
    if a.command == "freeze":
        return cmd_freeze()
    return cmd_update(a.skip_random)


if __name__ == "__main__":
    sys.exit(main())
