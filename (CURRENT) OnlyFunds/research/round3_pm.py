"""Round 3: position management and account-level risk (pre-registered in docs/research/ROUND3_PM_PROGRAMME.md).

Research only: no orders, no keys, no capital. Frozen T50 signals unchanged; the frozen simulators are not edited.
Outputs: docs/research/08_round3_pm.md, docs/research/round3_pm_results.json; results go to the hypothesis register.

    python research/round3_pm.py
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import hypotheses as H  # noqa: E402
import ml_lab as L  # noqa: E402
import ml_round1 as M  # noqa: E402
import portfolio_pm as PM  # noqa: E402
import round1 as R1  # noqa: E402
import t50_scrutiny as S  # noqa: E402
from portfolio import Costs, stats  # noqa: E402

OUT_MD = os.path.join(ROOT, "docs", "research", "08_round3_pm.md")
OUT_JSON = os.path.join(ROOT, "docs", "research", "round3_pm_results.json")
STRESS = Costs(0.004, 0.002, 0.005)
SPLITS = {"DEV": (L.MAIN, "2019-01-01", "2023-01-01"), "VAL": (L.MAIN, "2023-01-01", "2025-01-01"),
          "HOLDOUT": (L.HOLDOUT, "2019-01-01", "2025-01-01")}
BAND, LOCK = (0.15, 0.10), (2.0, 6.0)
CANDIDATES = {
    "R0": (PM.Rules(), False),
    "X1": (PM.Rules(trim_band=BAND, profit_lock=LOCK), False),
    "X2": (PM.Rules(trim_band=BAND, profit_lock=LOCK), True),
    "A-GROW": (PM.Rules(growth_trim=(3.0, 2.0)), False),
    "A-ATR": (PM.Rules(atr_cap=(0.0075, 0.005)), False),
    "A-ORISK": (PM.Rules(open_risk_cap=(0.10, 0.08)), False),
    "A-GROSS": (PM.Rules(gross_cap=(0.35, 0.30)), False),
    "A-CORR": (PM.Rules(corr_cap=(0.30, 0.25, 90)), False),
    "A-LOCK4": (PM.Rules(profit_lock=(4.0, 4.0)), False),
    "K1": (PM.Rules(trim_band=BAND, profit_lock=LOCK, open_risk_cap=(0.10, 0.08)), False),
    "K2": (PM.Rules(trim_band=BAND, profit_lock=LOCK, corr_cap=(0.30, 0.25, 90)), False),
    "K3": (PM.Rules(trim_band=BAND, profit_lock=LOCK, atr_cap=(0.0075, 0.005), open_risk_cap=(0.10, 0.08),
                    corr_cap=(0.30, 0.25, 90)), False),
}
N_BOOT = int(os.environ.get("R3_N_BOOT", "1000"))
BLOCK = 20
SMOKE = os.environ.get("R3_SMOKE")          # a directory: crash test on 3 coins per split, outputs there, no register
if SMOKE:
    OUT_MD, OUT_JSON = os.path.join(SMOKE, "smoke.md"), os.path.join(SMOKE, "smoke.json")
    SPLITS = {k: (["BTCUSDT"] + v[0][1:3], v[1], v[2]) for k, v in SPLITS.items()}


def positions(tr):
    if not len(tr):
        return pd.DataFrame(columns=["key", "pnl", "risk", "r"])
    g = tr.groupby(["pair", "entry_day"]).agg(pnl=("pnl", "sum"), risk=("initial_risk", "sum"))
    g["r"] = g.pnl / g.risk
    return g


def run_book(inp, btc_ok, start, end, rules, gate, no_halt, costs=Costs()):
    fr, en, ex, at = inp
    if gate:
        en = {p: s & btc_ok.reindex(s.index).fillna(False).astype(bool) for p, s in en.items()}
    return PM.simulate(fr, en, ex, at, start, end, M.conservative(no_halt), costs, capital=10000.0, rules=rules)


def metrics(res, risk, base_pos):
    s = stats(res, 10000.0)
    pos = positions(res.trades)
    tr = res.trades
    top5 = base_pos.pnl.nlargest(5) if base_pos is not None and len(base_pos) else pd.Series(dtype=float)
    kept = float(pos.pnl.reindex(top5.index).fillna(0).sum() / top5.sum()) if top5.sum() > 0 else np.nan
    eq = res.equity
    trims = tr[tr.reason.str.startswith("trim")].reason.value_counts().to_dict() if len(tr) else {}
    return {"return_pct": s["total_return_pct"], "cagr_pct": s["cagr_pct"], "max_dd_pct": s["max_dd_pct"],
            "ret_over_dd": s["total_return_pct"] / max(s["max_dd_pct"], 1e-9), "sharpe": s["sharpe"],
            "positions": int(len(pos)), "mean_R_per_position": float(pos.r.mean()) if len(pos) else np.nan,
            "total_R_positions": float(pos.r.sum()) if len(pos) else 0.0, "trims": trims,
            "profit_lock_exits": int((tr.reason == "profit_lock").sum()) if len(tr) else 0,
            "max_position_pct": float(risk.max_position_pct.max()), "max_gross_pct": float(risk.position_value_pct.max()),
            "max_open_risk_pct": float(risk.open_risk_to_stop_pct.max()),
            "max_corr_exposure_pct": float(risk.corr_exposure_pct.max()),
            "max_gap30_loss_pct": float(risk.gap30_loss_pct.max()),
            "worst_day_pct": float(eq.pct_change().min() * 100), "worst_5d_pct": float(eq.pct_change(5).min() * 100),
            "top5_R0_kept": kept, "halted_on": s["halted_on"]}


def _path_stats(r):
    eq = np.cumprod(1 + r)
    years = len(r) / 365.25
    return (eq[-1] ** (1 / years) - 1) * 100, (1 - (eq / np.maximum.accumulate(eq)).min()) * 100


def paired_bootstrap(eq_a, eq_b, seed=3303):
    """Circular block bootstrap, same blocks for both books. Returns 90% intervals of dCAGR, dMaxDD (b - a)."""
    ra = eq_a.pct_change().fillna(0).to_numpy()
    rb = eq_b.reindex(eq_a.index).pct_change().fillna(0).to_numpy()
    n = len(ra)
    rng = np.random.default_rng(seed)
    dc, dd = [], []
    for _ in range(N_BOOT):
        starts = rng.integers(0, n, int(np.ceil(n / BLOCK)))
        idx = (starts[:, None] + np.arange(BLOCK)[None, :]).ravel()[:n] % n
        ca, da = _path_stats(ra[idx])
        cb, db = _path_stats(rb[idx])
        dc.append(cb - ca)
        dd.append(db - da)
    q = lambda v: [float(np.percentile(v, 5)), float(np.percentile(v, 95)), float(np.mean(np.array(v) > 0))]
    return {"d_cagr_90ci_and_p_pos": q(dc), "d_maxdd_90ci_and_p_pos": q(dd)}


def eligible(x, b):
    return {"return_ge_80pct": x["return_pct"] >= (0.8 * b["return_pct"] if b["return_pct"] > 0 else b["return_pct"]),
            "top5_kept_70pct": bool(x["top5_R0_kept"] >= 0.7), "open_risk_le_15": x["max_open_risk_pct"] <= 15.0,
            "max_position_le_20": x["max_position_pct"] <= 20.0}


def confirm(x, b, xs, bs):
    return {"ret_dd_ge_R0": x["ret_over_dd"] >= b["ret_over_dd"], "max_dd_le_R0": x["max_dd_pct"] <= b["max_dd_pct"],
            **eligible(x, b), "stress_ret_dd_ge_R0": xs["ret_over_dd"] >= bs["ret_over_dd"]}


def main():
    t0 = time.time()
    data = L.load_universe(sorted({c for u, _, _ in SPLITS.values() for c in u} | {"BTCUSDT"}))
    btc_ok = R1.btc_filter(data["BTCUSDT"])
    ctx = {"btc_ok": btc_ok}
    cells, eq_store = {}, {}
    for split, (uni, a, b) in SPLITS.items():
        inp = S.portfolio_inputs({s: data[s] for s in uni if s in data}, ctx)
        for mode in ("no_halt", "as_configured"):
            costs_list = [("base", Costs())] + ([("stress", STRESS)] if split != "DEV" and mode == "no_halt" else [])
            for cname, costs in costs_list:
                base_pos = None
                for name, (rules, gate) in CANDIDATES.items():
                    res, risk = run_book(inp, btc_ok, a, b, rules, gate, mode == "no_halt", costs)
                    if name == "R0":
                        base_pos = positions(res.trades)
                    cells[(split, mode, cname, name)] = metrics(res, risk, base_pos)
                    if mode == "no_halt" and cname == "base":
                        eq_store[(split, name)] = res.equity
            print(f"{split} {mode} done ({time.time() - t0:.0f}s)", flush=True)
    get = lambda s, n, m="no_halt", c="base": cells[(s, m, c, n)]
    # selection on DEV
    elig = {n: eligible(get("DEV", n), get("DEV", "R0")) for n in CANDIDATES if n != "R0"}
    ok = [n for n, e in elig.items() if all(e.values())]
    selected = max(ok, key=lambda n: (get("DEV", n)["ret_over_dd"], -get("DEV", n)["max_dd_pct"])) if ok else None
    conf = {}
    for n in [x for x in CANDIDATES if x != "R0"]:
        conf[n] = {s: confirm(get(s, n), get(s, "R0"), get(s, n, c="stress"), get(s, "R0", c="stress"))
                   for s in ("VAL", "HOLDOUT")}
    passes = lambda n: all(all(v.values()) for v in conf[n].values())
    boot = {(s, n): paired_bootstrap(eq_store[(s, "R0")], eq_store[(s, n)]) for s in SPLITS for n in CANDIDATES if n != "R0"}
    verdict_sel = ("rejected (no eligible candidate on DEV)" if selected is None
                   else "supported" if passes(selected) else "rejected")
    verdict_k3 = "supported" if passes("K3") else "rejected"
    x2 = {s: get(s, "X2")["ret_over_dd"] > get(s, "X1")["ret_over_dd"] for s in ("VAL", "HOLDOUT")}
    out = {"generated_utc": pd.Timestamp.now(tz="UTC").isoformat(), "candidates": {n: {"rules": r.as_dict(), "btc_gate": g}
                                                                                for n, (r, g) in CANDIDATES.items()},
           "cells": [{"split": k[0], "mode": k[1], "costs": k[2], "candidate": k[3], **v} for k, v in cells.items()],
           "dev_eligibility": elig, "selected": selected, "confirmation": conf,
           "bootstrap": [{"split": k[0], "candidate": k[1], **v} for k, v in boot.items()],
           "verdicts": {"H3-PM-SEL": verdict_sel, "H3-K3": verdict_k3, "H3-X2": x2},
           "data_sha256": L.data_hashes(data), "runtime_s": time.time() - t0}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=str)
    write_md(out)
    if not SMOKE:
        ref = "docs/research/08_round3_pm.md; docs/research/round3_pm_results.json"
        s = get("VAL", selected) if selected else None
        H.record_result("H3-PM-SEL", verdict_sel.split(" ")[0],
                        f"selected on DEV: {selected}; " + ("; ".join(f"{sp}: " + ", ".join(k for k, v in conf[selected][sp].items() if not v)
                                                                    for sp in ("VAL", "HOLDOUT")) + " failed" if selected and not passes(selected)
                                                          else "all 7 checks passed on VAL and HOLDOUT" if selected else "none eligible"), ref)
        H.record_result("H3-K3", verdict_k3, "failed: " + "; ".join(f"{sp}: " + ", ".join(k for k, v in conf["K3"][sp].items() if not v)
                                                                      for sp in ("VAL", "HOLDOUT")) if not passes("K3") else "all 7 checks passed", ref)
        H.record_result("H3-X2", "supported" if all(x2.values()) else "rejected" if not any(x2.values()) else "inconclusive",
                        f"X2 ret/DD > X1: VAL {x2['VAL']}, HOLDOUT {x2['HOLDOUT']} (descriptive)", ref)
    print(f"selected {selected}; H3-PM-SEL {verdict_sel}; H3-K3 {verdict_k3}; X2>X1 {x2}; {time.time() - t0:.0f}s")


def write_md(o):
    c = pd.DataFrame(o["cells"])
    fmt = lambda v, p=1: "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{p}f}"
    L_ = ["# Round 3: position management and account-level risk (generated by `research/round3_pm.py`)\n",
          "Pre-registration: `ROUND3_PM_PROGRAMME.md` (committed before this run). Frozen T50 signals, single "
          "10,000 USDT account, Conservative limits, 0.5% initial risk, costs 0.20% fee / 0.05% slippage / 0.20% stop "
          "slippage. Historical splits were examined in Rounds 1–2, so any pass is **supported (historical)**, not "
          "proven. No forward results exist yet (first evaluation 2026-11-03).\n",
          f"**Selected on DEV:** {o['selected']}. **H3-PM-SEL:** {o['verdicts']['H3-PM-SEL']}. **H3-K3:** "
          f"{o['verdicts']['H3-K3']}. **X2 > X1 (ret/DD):** VAL {o['verdicts']['H3-X2']['VAL']}, HOLDOUT "
          f"{o['verdicts']['H3-X2']['HOLDOUT']}.\n"]
    for split in SPLITS:
        L_ += [f"## {split} (no halt, base costs)\n",
               "| book | return % | CAGR % | max DD % | ret/DD | Sharpe | positions | R/position | largest pos % | max gross % | max open risk % | max corr exp % | gap-30% loss % | worst day % | worst 5d % | top-5 kept | trims |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in c[(c.split == split) & (c["mode"] == "no_halt") & (c.costs == "base")].iterrows():
            tr = ", ".join(f"{k.replace('trim_', '').replace('trim', 'band')} {v}" for k, v in r.trims.items()) or "–"
            L_.append(f"| {r.candidate} | {r.return_pct:+.1f} | {r.cagr_pct:+.1f} | {r.max_dd_pct:.1f} | {r.ret_over_dd:.2f} | "
                      f"{fmt(r.sharpe, 2)} | {r.positions} | {fmt(r.mean_R_per_position, 2)} | {r.max_position_pct:.1f} | "
                      f"{r.max_gross_pct:.1f} | {r.max_open_risk_pct:.1f} | {r.max_corr_exposure_pct:.1f} | "
                      f"{r.max_gap30_loss_pct:.1f} | {r.worst_day_pct:+.1f} | {r.worst_5d_pct:+.1f} | {fmt(r.top5_R0_kept, 2)} | {tr} |")
        L_.append("")
    L_ += ["## As configured (10% kill switch, permanent): return % / max DD % / kill-switch date\n",
           "| book | " + " | ".join(SPLITS) + " |", "|---|" + "---|" * len(SPLITS)]
    for n in o["candidates"]:
        row = []
        for split in SPLITS:
            r = c[(c.split == split) & (c["mode"] == "as_configured") & (c.candidate == n)].iloc[0]
            row.append(f"{r.return_pct:+.1f} / {r.max_dd_pct:.1f} / {r.halted_on}")
        L_.append(f"| {n} | " + " | ".join(row) + " |")
    L_ += ["\n## Stress costs (0.40% fee, 0.20% slippage, 0.50% stop slippage), no halt: ret/DD\n",
           "| book | VAL | HOLDOUT |", "|---|---|---|"]
    for n in o["candidates"]:
        v = [c[(c.split == s) & (c.costs == "stress") & (c.candidate == n)].iloc[0].ret_over_dd for s in ("VAL", "HOLDOUT")]
        L_.append(f"| {n} | {v[0]:.2f} | {v[1]:.2f} |")
    L_ += ["\n## DEV eligibility and VAL/HOLDOUT confirmation checks\n",
           "| candidate | DEV eligible | failed DEV checks | VAL failed checks | HOLDOUT failed checks |", "|---|---|---|---|---|"]
    for n, e in o["dev_eligibility"].items():
        fails = lambda d: ", ".join(k for k, v in d.items() if not v) or "none"
        L_.append(f"| {n} | {'yes' if all(e.values()) else 'no'} | {fails(e)} | {fails(o['confirmation'][n]['VAL'])} | "
                  f"{fails(o['confirmation'][n]['HOLDOUT'])} |")
    L_ += ["\n## Uncertainty: paired block bootstrap against R0 (no halt; 20-day blocks; 90% intervals; P(Δ > 0))\n",
           "| split | candidate | Δ CAGR pts | P(ΔCAGR>0) | Δ max DD pts | P(ΔDD>0) |", "|---|---|---|---|---|---|"]
    for b in o["bootstrap"]:
        dc, dd = b["d_cagr_90ci_and_p_pos"], b["d_maxdd_90ci_and_p_pos"]
        L_.append(f"| {b['split']} | {b['candidate']} | {dc[0]:+.1f} to {dc[1]:+.1f} | {dc[2]:.2f} | {dd[0]:+.1f} to {dd[1]:+.1f} | {dd[2]:.2f} |")
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L_) + "\n")


if __name__ == "__main__":
    main()
