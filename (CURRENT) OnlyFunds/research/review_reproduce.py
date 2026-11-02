"""Independent reproduction of the headline research figures (SAFETY_AND_MONITORING_REVIEW_2026-11-02.md, section 4).

Read-only: it changes no published result and writes nothing unless an output path is given. Run from the app
folder; uses committed code, the committed 2019-2024 data and the saved Round-1 models (no retraining). Where
possible a figure is recomputed by a different route than the one that published it (the frozen simulators, the
frozen forward tracker's engine, or the raw trade log and closing prices).

    python research/review_reproduce.py [output.json]

Note: on the last day of a split, the simulator records risk before its forced end-of-period close, so the trade-log
stop-risk recomputation differs on that one day only; the comparison below excludes it.
"""
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(APP)
sys.path[:0] = [APP, os.path.join(APP, "research")]
import ml_lab as L  # noqa: E402
import ml_round1 as M  # noqa: E402
import portfolio as P  # noqa: E402
import portfolio_pm as PM  # noqa: E402
import portfolio_variants as PV  # noqa: E402
import round1 as R1  # noqa: E402
import t50_scrutiny as S  # noqa: E402
import forward_shadow as FS  # noqa: E402
from portfolio import Costs, stats  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else None
t0 = time.time()
pub1 = json.load(open("docs/research/ml_round1_results.json", encoding="utf-8"))
r3 = pd.DataFrame(json.load(open("docs/research/round3_pm_results.json", encoding="utf-8"))["cells"])
rc = json.load(open("docs/research/reconciliation.json", encoding="utf-8"))
data = L.load_universe(L.UNIVERSE)
ctx = {"btc_ok": R1.btc_filter(data["BTCUSDT"])}
out = {}
SPL = {"DEV": (L.MAIN, "2019-01-01", "2023-01-01"), "VAL": (L.MAIN, "2023-01-01", "2025-01-01"),
       "HOLDOUT": (L.HOLDOUT, "2019-01-01", "2025-01-01")}


def pub(split, cand, mode="no_halt", costs="base"):
    return r3[(r3.split == split) & (r3["mode"] == mode) & (r3.costs == costs) & (r3.candidate == cand)].iloc[0]


def log(k, v):
    out[k] = v
    print(k, json.dumps(v, default=str)[:600], f"({time.time() - t0:.0f}s)", flush=True)


# 1) +251.2R: research route, and the frozen forward tracker's own per-coin engine as a second route
t = M.trades_of(M.run_t50(data, ctx, M.cfg(), "2021-01-01", "2025-01-01"))
fs_cfg = R1.config()
fs_cfg.close_at_end = M.cfg().close_at_end
old = FS.WARMUP_START
FS.WARMUP_START = "2019-01-01"
ft, _, fop = FS.signal_book("T50", data, ctx, fs_cfg, "2021-01-01", "2025-01-01")
FS.WARMUP_START = old
ft = ft.assign(entry_ts=pd.to_datetime(ft.entry_time, utc=True))
ft = ft[(ft.entry_ts >= "2021-01-01") & (ft.entry_ts < "2025-01-01")]
log("1_G0_251R", {"published_trades": pub1["base_G0"]["trades"], "published_R": pub1["base_G0"]["total_R"],
                  "research_route_trades": int(len(t)), "research_route_R": float(t.r_multiple.sum()),
                  "shadow_engine_trades": int(len(ft)), "shadow_engine_R": float(ft.r_multiple.sum()),
                  "close_at_end": bool(fs_cfg.close_at_end)})

# 2) 188.9R GM2: saved Round-1 M2 models (no retraining), gate p >= 0.5, same per-coin engine
panel = L.build_panel(data)
kind = L.MODEL_SPECS["M2"][0]
preds = []
for year in L.TEST_YEARS:
    tr, te = L.fold_masks(panel["day"], panel["label_end"], year)
    m = joblib.load(os.path.join(APP, "state", "ml", "models", f"M2_{year}.pkl"))
    x = panel[te].copy()
    x["p_M2"] = L.predict(kind, m, x[L.FEATURES])
    preds.append(x[["pair", "day", "p_M2"]])
oos = pd.concat(preds, ignore_index=True)
sdays = []
for pair, df in data.items():
    sig = R1.signals("T50", df, {**ctx, "pair": pair})
    ts = pd.to_datetime(df["timestamp"], utc=True)
    sdays += [(pair, d) for d in ts[sig["entry"].to_numpy(bool)]]
sd = pd.DataFrame(sdays, columns=["pair", "day"])
sd = sd[(sd.day >= pd.Timestamp("2021-01-01", tz="UTC")) & (sd.day < pd.Timestamp("2025-01-01", tz="UTC"))]
v = oos.set_index(["pair", "day"])["p_M2"].reindex(pd.MultiIndex.from_frame(sd)).to_numpy()
gate = M._to_gate(sd, (~np.isfinite(v)) | (v >= 0.5))
tg = M.trades_of(M.run_t50(data, ctx, M.cfg(), "2021-01-01", "2025-01-01", gate))
g = pub1["gates"]["GM2"]
log("2_GM2_188.9R", {"published_trades": g["g_trades"], "published_R": g["g_total_R"],
                     "published_blocked": g["signal_days_blocked"], "recomputed_blocked": M.blocked_count(gate),
                     "recomputed_trades": int(len(tg)), "recomputed_R": float(tg.r_multiple.sum()),
                     "recomputed_delta_vs_G0": float(tg.r_multiple.sum() - t.r_multiple.sum())})


# 3) shared account, all 25 coins 2021-24, frozen single-account simulator, no halt
def inputs(uni):
    return S.portfolio_inputs({s: data[s] for s in uni if s in data}, ctx)


fr, en, ex, at = inputs(L.UNIVERSE)
res = P.simulate(fr, en, ex, at, "2021-01-01", "2025-01-01", M.conservative(True), Costs(), capital=10000.0)
tr_ = res.trades
posR = tr_.groupby(["pair", "entry_day"]).apply(lambda g: g.pnl.sum() / g.initial_risk.sum())
s = stats(res, 10000.0)
a = rc["splits"]["ALL25_2021_24"]["R0_no_halt"]
log("3_shared_account_41R", {"published_R": a["book_total_R"], "published_return": a["return_pct"],
                             "published_dd": a["max_dd_pct"], "published_positions": a["positions"],
                             "recomputed_R": float(posR.sum()), "recomputed_return": s["total_return_pct"],
                             "recomputed_dd": s["max_dd_pct"], "recomputed_positions": int(len(posR)),
                             "kill_switch_as_configured": stats(P.simulate(fr, en, ex, at, "2021-01-01", "2025-01-01",
                                                                           M.conservative(False), Costs(),
                                                                           capital=10000.0), 10000.0)["halted_on"]})


# 4) open risk to stops, recomputed from the trade log and closes only (not from the simulator's risk columns)
def open_risk_from_trades(res, frames, fee=0.0, stop_slip=0.0):
    """At each close: sum over held positions of max(qty_held x (close - stop), 0) / equity.
    With fee/stop_slip > 0 it also adds the exit costs of a stop fill (qty x stop x (1 - (1-ss)(1-fee)))."""
    tr = res.trades.copy()
    key = "position_id" if "position_id" in tr else None
    if key is None:
        tr["position_id"] = tr.groupby(["pair", "entry_day"]).ngroup()
    closes = pd.DataFrame({p: f["Close"] for p, f in frames.items()}).ffill()
    eq = res.equity
    rows = []
    for pid, g in tr.groupby("position_id"):
        q0 = g.qty.sum()
        sold = g.groupby("exit_day").qty.sum().cumsum()
        rows.append((g.pair.iloc[0], g.entry_day.iloc[0], g.exit_day.max(), q0, sold, float(g.stop.iloc[0]),
                     bool((g.stop != g.stop.iloc[0]).any())))
    risk = pd.Series(0.0, index=eq.index)
    for pair, ed, xd, q0, sold, stop, moved in rows:
        days = eq.index[(eq.index >= ed) & (eq.index < xd + pd.Timedelta(days=1))]
        held = q0 - sold.reindex(days, method="ffill").fillna(0.0).to_numpy()
        c = closes[pair].reindex(days).to_numpy()
        risk.loc[days] += np.maximum(held * (c - stop), 0) + held * stop * (1 - (1 - stop_slip) * (1 - fee))
    return risk / eq * 100


for split in ("DEV", "VAL", "HOLDOUT"):
    uni, a_, b_ = SPL[split]
    fr, en, ex, at = inputs(uni)
    res = P.simulate(fr, en, ex, at, a_, b_, M.conservative(True), Costs(), capital=10000.0)
    orisk = open_risk_from_trades(res, fr)
    orisk_cost = open_risk_from_trades(res, fr, fee=0.002, stop_slip=0.002)
    p = pub(split, "R0")
    log(f"4_R0_open_risk_{split}", {"published_max_open_risk": p.max_open_risk_pct,
                                    "recomputed_from_trades": float(orisk.max()),
                                    "with_exit_fee_and_stop_slippage": float(orisk_cost.max()),
                                    "published_dd": p.max_dd_pct, "recomputed_dd": stats(res, 10000.0)["max_dd_pct"]})

# 5) the 10% / 8% open-risk cap (A-ORISK in Round 3 = B-ORISK in Round 4)
for split in ("DEV", "VAL", "HOLDOUT"):
    uni, a_, b_ = SPL[split]
    fr, en, ex, at = inputs(uni)
    res, risk = PM.simulate(fr, en, ex, at, a_, b_, M.conservative(True), Costs(), capital=10000.0,
                            rules=PM.Rules(open_risk_cap=(0.10, 0.08)))
    st = stats(res, 10000.0)
    orisk = open_risk_from_trades(res, fr)
    orisk_cost = open_risk_from_trades(res, fr, fee=0.002, stop_slip=0.002)
    above = orisk[orisk > 10.0]
    tr = res.trades
    trims = tr[tr.reason == "trim_open_risk"]
    # a breach at close d should be followed by an open-risk trim at the next open (d + 1 day)
    nxt = set(pd.to_datetime(trims.exit_day).dt.normalize())
    followed = [d for d in above.index if (d + pd.Timedelta(days=1)).normalize() in nxt]
    consec = int(((orisk > 10.0) & (orisk.shift(1) > 10.0)).sum())
    stops = tr[tr.reason == "stop_loss"]
    gap = stops[stops.exit_raw < stops.stop - 1e-12]
    # missing bars for held coins (a position whose coin has no bar is not counted by the cap's sum)
    miss = 0
    for (pair, ed), g in tr.groupby(["pair", "entry_day"]):
        days = res.equity.index[(res.equity.index >= ed) & (res.equity.index < g.exit_day.max())]
        miss += int((~days.isin(fr[pair].index)).sum())
    p = pub(split, "A-ORISK")
    pst = pub(split, "A-ORISK", costs="stress") if split != "DEV" else None
    log(f"5_cap_{split}", {
        "published_return": p.return_pct, "published_dd": p.max_dd_pct, "published_ret_over_dd": p.ret_over_dd,
        "recomputed_return": st["total_return_pct"], "recomputed_dd": st["max_dd_pct"],
        "recomputed_ret_over_dd": st["total_return_pct"] / st["max_dd_pct"],
        "risk_column_vs_trade_log_max_abs_diff_pts": float((risk.open_risk_to_stop_pct - orisk).abs().iloc[:-1].max()),
        "max_open_risk_pct": float(orisk.max()), "max_open_risk_incl_exit_costs_pct": float(orisk_cost.max()),
        "days": int(len(orisk)), "days_above_10pct": int(len(above)), "days_above_8pct": int((orisk > 8).sum()),
        "breaches_followed_by_trim_next_open": int(len(followed)), "consecutive_breach_days": consec,
        "worst_breach_days": {str(d.date()): round(float(v), 2) for d, v in above.nlargest(5).items()},
        "open_risk_trims": int(len(trims)),
        "stop_exits": int(len(stops)), "stop_exits_gapped_through_stop": int(len(gap)),
        "worst_gap_stop_R": float(gap.r.min()) if len(gap) else None,
        "worst_stop_R": float(stops.r.min()) if len(stops) else None,
        "held_days_without_a_bar": miss,
        "stress_published_ret_over_dd": None if pst is None else pst.ret_over_dd})

# 6) X1 vs X2 with the frozen experimental simulator (portfolio_variants), VAL and HOLDOUT, no halt
for split in ("VAL", "HOLDOUT"):
    uni, a_, b_ = SPL[split]
    fr, en, ex, at = inputs(uni)
    eg = {p_: s_ & ctx["btc_ok"].reindex(s_.index).fillna(False).astype(bool) for p_, s_ in en.items()}
    row = {}
    for name, e in (("X1", en), ("X2", eg)):
        res, _ = PV.simulate(fr, e, ex, at, a_, b_, M.conservative(True), Costs(), capital=10000.0,
                             trim_band=(0.15, 0.10), profit_lock=(2.0, 6.0))
        st = stats(res, 10000.0)
        p = pub(split, name)
        row[name] = {"published_ret_over_dd": p.ret_over_dd, "recomputed_ret_over_dd": st["total_return_pct"] / st["max_dd_pct"],
                     "published_return": p.return_pct, "recomputed_return": st["total_return_pct"],
                     "published_dd": p.max_dd_pct, "recomputed_dd": st["max_dd_pct"]}
    log(f"6_X1_vs_X2_{split}", row)

out["runtime_s"] = time.time() - t0
if OUT:
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=str)
    print("written", OUT)
