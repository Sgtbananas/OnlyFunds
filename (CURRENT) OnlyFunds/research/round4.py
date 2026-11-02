"""Round 4: regime engine and portfolio intelligence engine (pre-registered in docs/research/ROUND4_PROGRAMME.md).

Research only: no orders, no keys, no capital. Frozen T50 and RS3 signals unchanged; earlier simulators untouched.
Outputs: docs/research/09_round4.md, docs/research/round4_results.json; results go to the hypothesis register.

    python research/round4.py
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
import portfolio_engine as PE  # noqa: E402
import regime_engine as RE  # noqa: E402
import round1 as R1  # noqa: E402
import round3_pm as R3  # noqa: E402
import t50_scrutiny as S  # noqa: E402
from portfolio import Costs  # noqa: E402

OUT_MD = os.path.join(ROOT, "docs", "research", "09_round4.md")
OUT_JSON = os.path.join(ROOT, "docs", "research", "round4_results.json")
SPLITS = R3.SPLITS
STRESS = R3.STRESS
ORISK = (0.10, 0.08)
OFF_CAP = (0.05, 0.04)
REPL = (20, 0.5)
SMOKE = os.environ.get("R4_SMOKE")
if SMOKE:
    OUT_MD, OUT_JSON = os.path.join(SMOKE, "smoke.md"), os.path.join(SMOKE, "smoke.json")
    SPLITS = {k: (["BTCUSDT"] + v[0][1:4], v[1], v[2]) for k, v in SPLITS.items()}


def split_inputs(data, uni, btc_ok):
    d = {s: data[s] for s in uni if s in data}
    fr, en, ex, at = S.portfolio_inputs(d, {"btc_ok": btc_ok})
    member, _ = R1.rs_basket(d, btc_ok)
    rs_en, rs_ex = {}, {}
    for pair, df in d.items():
        sig = R1.signals("RS3", df, {"btc_ok": btc_ok, "member": member, "pair": pair})
        idx = pd.to_datetime(df["timestamp"], utc=True)
        rs_en[pair] = pd.Series(sig["entry"].to_numpy(bool), index=idx)
        rs_ex[pair] = pd.Series(sig["exit"].to_numpy(bool), index=idx)
    rr = L.rs_rank(d)
    rank = {p: rr[p] for p in d}
    return {"fr": fr, "t50": (en, ex), "rs3": (rs_en, rs_ex), "atr": at, "rank": rank}


def books(rule_off, rule_off_m, model_off):
    Rk = PE.Rules(open_risk_cap=ORISK)
    off_rules = PE.Rules(open_risk_cap=ORISK, open_risk_cap_off=OFF_CAP)
    return {
        "R0": dict(strats=("t50",)),
        "B-ORISK": dict(strats=("t50",), rules=Rk),
        "RS3B": dict(strats=("rs3",)),
        "E1": dict(strats=("t50",), rank=True),
        "E2": dict(strats=("t50",), rank=True, replace=REPL),
        "E3": dict(strats=("t50",), rules=off_rules, risk_off=rule_off),
        "E3m": dict(strats=("t50",), rules=off_rules, risk_off=rule_off_m),
        "E4": dict(strats=("t50",), rules=off_rules, risk_off=model_off),
        "E5": dict(strats=("t50",), rules=PE.Rules(open_risk_cap=ORISK, corr_risk_cap=(0.08, 0.065, 90))),
        "PIE": dict(strats=("t50", "rs3"), rules=off_rules, rank=True, replace=REPL, risk_off=rule_off),
        "PIE-ML": dict(strats=("t50", "rs3"), rules=off_rules, rank=True, replace=REPL, risk_off=model_off),
    }


def run_book(inp, spec, start, end, no_halt, costs):
    strategies = {s.upper(): inp[s] for s in spec["strats"]}
    first = inp[spec["strats"][0]]
    return PE.simulate(inp["fr"], first[0], first[1], inp["atr"], start, end, M.conservative(no_halt), costs,
                       capital=10000.0, rules=spec.get("rules", PE.Rules()), strategies=strategies,
                       rank=inp["rank"] if spec.get("rank") else None, replace=spec.get("replace"),
                       risk_off=spec.get("risk_off"))


def improves(get, x, b):
    base = {s: get(s, x)["ret_over_dd"] > get(s, b)["ret_over_dd"] and get(s, x)["return_pct"] >= get(s, b)["return_pct"]
            for s in SPLITS}
    stress = {s: get(s, x, c="stress")["ret_over_dd"] >= get(s, b, c="stress")["ret_over_dd"] for s in ("VAL", "HOLDOUT")}
    if all(base.values()) and all(stress.values()):
        v = "supported"
    elif base.get("HOLDOUT") and sum(base.values()) >= 2:
        v = "inconclusive"
    else:
        v = "rejected"
    return v, {"base": base, "stress": stress}


def main():
    t0 = time.time()
    data = L.load_universe(sorted({c for u, _, _ in SPLITS.values() for c in u} | set(L.MAIN)))
    btc_ok = R1.btc_filter(data["BTCUSDT"])
    # ---------------- Part A: regime engine
    m = RE.frame(data)
    oos = RE.walk_forward(m)
    ev = RE.evaluate(oos)
    reg_verdicts = {t: RE.verdict(ev, t) for t in RE.TARGETS}
    calib = {t: RE.calibration(oos, t).reset_index(drop=True).to_dict("records") for t in RE.TARGETS if t != "direction"}
    model_off = RE.risk_off_model(oos)
    c = data["BTCUSDT"].set_index(pd.to_datetime(data["BTCUSDT"]["timestamp"], utc=True))["Close"]
    sma = c.rolling(100).mean()
    rule_off = (c <= sma) & sma.notna()
    rule_off_m = rule_off & pd.Series(rule_off.index.isin(model_off.index), index=rule_off.index)
    overlap = {"model_off_days": int(model_off.sum()), "rule_off_days_2021_24": int(rule_off_m.sum()),
               "both": int((model_off & rule_off.reindex(model_off.index)).sum()),
               "model_days": int(len(model_off))}
    print(f"regime engine done ({time.time() - t0:.0f}s): {reg_verdicts}", flush=True)
    # ---------------- Part B: books
    B = books(rule_off, rule_off_m, model_off)
    cells, eq = {}, {}
    for split, (uni, a, b) in SPLITS.items():
        inp = split_inputs(data, uni, btc_ok)
        for mode in ("no_halt", "as_configured"):
            for cname, costs in [("base", Costs())] + ([("stress", STRESS)] if split != "DEV" and mode == "no_halt" else []):
                base_pos = None
                for name, spec in B.items():
                    res, risk = run_book(inp, spec, a, b, mode == "no_halt", costs)
                    if name == "R0":
                        base_pos = R3.positions(res.trades)
                    met = R3.metrics(res, risk, base_pos)
                    tr = res.trades
                    met["replaced"] = int((tr.reason == "replaced").sum()) if len(tr) else 0
                    met["by_strategy_positions"] = {}
                    cells[(split, mode, cname, name)] = met
                    if mode == "no_halt" and cname == "base":
                        eq[(split, name)] = res.equity
            print(f"{split} {mode} done ({time.time() - t0:.0f}s)", flush=True)
    get = lambda s, n, m_="no_halt", c="base": cells[(s, m_, c, n)]
    hyp = {"H4-SLOT": improves(get, "E1", "R0"), "H4-REPL": improves(get, "E2", "E1"),
           "H4-RISKOFF-RULE": improves(get, "E3", "B-ORISK"), "H4-RISKOFF-ML": improves(get, "E4", "E3m"),
           "H4-CORR": improves(get, "E5", "B-ORISK")}
    pie = {s: R3.confirm(get(s, "PIE"), get(s, "R0"), get(s, "PIE", c="stress"), get(s, "R0", c="stress"))
           for s in ("VAL", "HOLDOUT")}
    pie_dev = get("DEV", "PIE")["ret_over_dd"] >= get("DEV", "R0")["ret_over_dd"]
    hyp["H4-PIE"] = ("supported" if pie_dev and all(all(v.values()) for v in pie.values()) else "rejected",
                     {"dev_ret_dd_ge_R0": pie_dev, **pie})
    boot = {}
    for x, bse in (("E1", "R0"), ("E2", "E1"), ("E3", "B-ORISK"), ("E4", "E3m"), ("E5", "B-ORISK"), ("PIE", "R0"),
                   ("PIE", "B-ORISK"), ("B-ORISK", "R0")):
        for s in SPLITS:
            boot[f"{s}|{x}|{bse}"] = R3.paired_bootstrap(eq[(s, bse)], eq[(s, x)])
    out = {"generated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
           "regime": {"evaluation": ev.to_dict("records"), "verdicts": reg_verdicts, "calibration": calib,
                      "risk_off_overlap": overlap},
           "cells": [{"split": k[0], "mode": k[1], "costs": k[2], "book": k[3], **v} for k, v in cells.items()],
           "hypotheses": {k: {"verdict": v[0], "checks": v[1]} for k, v in hyp.items()},
           "bootstrap": boot, "data_sha256": L.data_hashes(data), "runtime_s": time.time() - t0}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=str)
    write_md(out)
    if not SMOKE:
        ref = "docs/research/09_round4.md; docs/research/round4_results.json"
        for t, v in reg_verdicts.items():
            e = ev[(ev.target == t) & (ev.span == "2021-24")]
            summ = (f"gain vs best baseline ({e.best_baseline.iloc[0]}) {e.gain_vs_best.iloc[0]:+.4f} nats "
                    f"(90% {e.gain_lo.iloc[0]:+.4f} to {e.gain_hi.iloc[0]:+.4f})") if len(e) else "no predictions"
            H.record_result(f"H4-REG-{t}", v.split(" ")[0], summ, ref)
        for k, (v, chk) in hyp.items():
            H.record_result(k, v, json.dumps(chk, default=str)[:400], ref)
    print({k: v[0] for k, v in hyp.items()}, f"{time.time() - t0:.0f}s")


def write_md(o):
    c = pd.DataFrame(o["cells"])
    fmt = lambda v, p=2: "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{p}f}"
    L_ = ["# Round 4: regime engine and portfolio intelligence engine (generated by `research/round4.py`)\n",
          "Pre-registration: `ROUND4_PROGRAMME.md` (committed before this run). Historical data only (2019–24, "
          "already examined): any pass is **supported (historical)**, never proven. Single 10,000 USDT account, "
          "Conservative limits, 0.5% initial risk (risk-off only lowers it), 3 positions.\n",
          "## Verdicts\n", "| hypothesis | verdict |", "|---|---|"]
    for t, v in o["regime"]["verdicts"].items():
        L_.append(f"| H4-REG-{t} | {v} |")
    for k, v in o["hypotheses"].items():
        L_.append(f"| {k} | {v['verdict']} |")
    L_ += ["\n## A. Regime engine: out-of-sample log loss (lower is better), walk-forward 2021–24\n",
           "| target | span | days | model | climatology | rule | best baseline | gain vs best (90% CI) | skill % | mean confidence |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in o["regime"]["evaluation"]:
        L_.append(f"| {r['target']} | {r['span']} | {r['days']} | {r['ll_model']:.3f} | {r['ll_clim']:.3f} | "
                  f"{r['ll_rule']:.3f} | {r['best_baseline']} | {r['gain_vs_best']:+.3f} ({r['gain_lo']:+.3f} to "
                  f"{r['gain_hi']:+.3f}) | {r['skill_vs_best_pct']:+.1f} | {r['mean_confidence_model']:.2f} |")
    L_ += ["\n### Calibration (model probability quintiles → observed frequency)\n"]
    for t, rows in o["regime"]["calibration"].items():
        L_.append(f"- **{t}:** " + "; ".join(f"{r['predicted']:.2f}→{r['observed']:.2f} (n {r['days']})" for r in rows))
    ov = o["regime"]["risk_off_overlap"]
    L_.append(f"\nRisk-off days 2021–24: model {ov['model_off_days']} of {ov['model_days']}, rule "
              f"{ov['rule_off_days_2021_24']}, both {ov['both']}.\n")
    for split in SPLITS:
        L_ += [f"## {split} (no halt, base costs)\n",
               "| book | return % | max DD % | ret/DD | Sharpe | positions | R/position | replaced | largest pos % | max open risk % | worst 5d % | top-5 R0 kept |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in c[(c.split == split) & (c["mode"] == "no_halt") & (c.costs == "base")].iterrows():
            L_.append(f"| {r.book} | {r.return_pct:+.1f} | {r.max_dd_pct:.1f} | {r.ret_over_dd:.2f} | {fmt(r.sharpe)} | "
                      f"{r.positions} | {fmt(r.mean_R_per_position)} | {r.replaced} | {r.max_position_pct:.1f} | "
                      f"{r.max_open_risk_pct:.1f} | {r.worst_5d_pct:+.1f} | {fmt(r.top5_R0_kept)} |")
        L_.append("")
    L_ += ["## As configured (10% kill switch): return % / max DD % / kill switch\n",
           "| book | " + " | ".join(SPLITS) + " |", "|---|" + "---|" * len(SPLITS)]
    for n in c.book.unique():
        row = []
        for split in SPLITS:
            r = c[(c.split == split) & (c["mode"] == "as_configured") & (c.book == n)].iloc[0]
            row.append(f"{r.return_pct:+.1f} / {r.max_dd_pct:.1f} / {r.halted_on}")
        L_.append(f"| {n} | " + " | ".join(row) + " |")
    L_ += ["\n## Stress costs, no halt: return % / ret/DD\n", "| book | VAL | HOLDOUT |", "|---|---|---|"]
    for n in c.book.unique():
        v = [c[(c.split == s) & (c.costs == "stress") & (c.book == n)].iloc[0] for s in ("VAL", "HOLDOUT")]
        L_.append(f"| {n} | {v[0].return_pct:+.1f} / {v[0].ret_over_dd:.2f} | {v[1].return_pct:+.1f} / {v[1].ret_over_dd:.2f} |")
    L_ += ["\n## Hypothesis checks\n"]
    for k, v in o["hypotheses"].items():
        L_.append(f"- **{k}: {v['verdict']}**. `{json.dumps(v['checks'], default=str)}`")
    L_ += ["\n## Paired block bootstrap (no halt; 90% intervals; P(Δ>0))\n",
           "| split | book vs comparator | Δ CAGR pts | P | Δ max DD pts | P |", "|---|---|---|---|---|---|"]
    for k, b in o["bootstrap"].items():
        s, x, bse = k.split("|")
        dc, dd = b["d_cagr_90ci_and_p_pos"], b["d_maxdd_90ci_and_p_pos"]
        L_.append(f"| {s} | {x} vs {bse} | {dc[0]:+.1f} to {dc[1]:+.1f} | {dc[2]:.2f} | {dd[0]:+.1f} to {dd[1]:+.1f} | {dd[2]:.2f} |")
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L_) + "\n")


if __name__ == "__main__":
    main()
