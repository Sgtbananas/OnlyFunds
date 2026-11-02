"""Re-run every published single-account cell and compare it with the committed results files. Read-only.

Covers Round 3 (docs/research/round3_pm_results.json: every candidate, split, halt mode and cost setting),
Round 4 (docs/research/round4_results.json: every book, including the regime-driven ones) and the reconciliation
(docs/research/reconciliation.json: R0 and X1 risk profiles per split, and the published risk-variant cells).
Bootstraps and hypothesis verdicts are not re-run; they are functions of these cells and fixed seeds.

    python research/review_published_unchanged.py
"""
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(APP)
sys.path[:0] = [APP, os.path.join(APP, "research")]
import ml_lab as L  # noqa: E402
import reconcile as RC  # noqa: E402
import regime_engine as RE  # noqa: E402
import round1 as R1  # noqa: E402
import round3_pm as R3  # noqa: E402
import round4 as R4  # noqa: E402
import t50_scrutiny as S  # noqa: E402
from portfolio import Costs, stats  # noqa: E402

t0 = time.time()
worst = {"diff": 0.0, "where": None, "fields": 0}
mismatches = []


def compare(where, pub, new):
    for k, v in pub.items():
        if k not in new:
            continue
        w = new[k]
        if isinstance(v, (int, float)) and not isinstance(v, bool) and isinstance(w, (int, float, np.floating, np.integer)):
            if (v is None or (isinstance(v, float) and math.isnan(v))) and (w is None or math.isnan(float(w))):
                continue
            d = abs(float(v) - float(w))
            worst["fields"] += 1
            if d > worst["diff"]:
                worst.update(diff=d, where=f"{where} {k}")
            if not np.isclose(float(v), float(w), rtol=1e-12, atol=1e-12):
                mismatches.append(f"{where} {k}: published {v} recomputed {w}")
        elif isinstance(v, (str, bool)) and str(v) != str(w):
            mismatches.append(f"{where} {k}: published {v!r} recomputed {w!r}")


# ---------------- Round 3
r3 = json.load(open("docs/research/round3_pm_results.json", encoding="utf-8"))["cells"]
data = L.load_universe(sorted({c for u, _, _ in R3.SPLITS.values() for c in u} | {"BTCUSDT"}))
btc_ok = R1.btc_filter(data["BTCUSDT"])
n3 = 0
for split, (uni, a, b) in R3.SPLITS.items():
    inp = S.portfolio_inputs({s: data[s] for s in uni if s in data}, {"btc_ok": btc_ok})
    for mode in ("no_halt", "as_configured"):
        for cname, costs in [("base", Costs())] + ([("stress", R3.STRESS)] if split != "DEV" and mode == "no_halt" else []):
            base_pos = None
            for name, (rules, gate) in R3.CANDIDATES.items():
                res, risk = R3.run_book(inp, btc_ok, a, b, rules, gate, mode == "no_halt", costs)
                if name == "R0":
                    base_pos = R3.positions(res.trades)
                met = R3.metrics(res, risk, base_pos)
                pub = [c for c in r3 if (c["split"], c["mode"], c["costs"], c["candidate"]) == (split, mode, cname, name)]
                if pub:
                    compare(f"R3 {split} {mode} {cname} {name}", pub[0], met)
                    n3 += 1
print(f"Round 3: {n3} cells compared ({time.time() - t0:.0f}s)", flush=True)

# ---------------- Round 4
r4 = json.load(open("docs/research/round4_results.json", encoding="utf-8"))["cells"]
data4 = L.load_universe(sorted({c for u, _, _ in R4.SPLITS.values() for c in u} | set(L.MAIN)))
btc4 = R1.btc_filter(data4["BTCUSDT"])
oos = RE.walk_forward(RE.frame(data4))
model_off = RE.risk_off_model(oos)
c = data4["BTCUSDT"].set_index(pd.to_datetime(data4["BTCUSDT"]["timestamp"], utc=True))["Close"]
sma = c.rolling(100).mean()
rule_off = (c <= sma) & sma.notna()
rule_off_m = rule_off & pd.Series(rule_off.index.isin(model_off.index), index=rule_off.index)
B = R4.books(rule_off, rule_off_m, model_off)
n4 = 0
for split, (uni, a, b) in R4.SPLITS.items():
    inp = R4.split_inputs(data4, uni, btc4)
    for mode in ("no_halt", "as_configured"):
        for cname, costs in [("base", Costs())] + ([("stress", R4.STRESS)] if split != "DEV" and mode == "no_halt" else []):
            base_pos = None
            for name, spec in B.items():
                res, risk = R4.run_book(inp, spec, a, b, mode == "no_halt", costs)
                if name == "R0":
                    base_pos = R3.positions(res.trades)
                met = R3.metrics(res, risk, base_pos)
                met["replaced"] = int((res.trades.reason == "replaced").sum()) if len(res.trades) else 0
                assert bool(risk["risk_reliable"].all()), f"{split} {name}: an unpriced held position occurred"
                pub = [x for x in r4 if (x["split"], x["mode"], x["costs"], x["book"]) == (split, mode, cname, name)]
                if pub:
                    compare(f"R4 {split} {mode} {cname} {name}", pub[0], met)
                    n4 += 1
print(f"Round 4: {n4} cells compared ({time.time() - t0:.0f}s)", flush=True)

# ---------------- reconciliation
rc = json.load(open("docs/research/reconciliation.json", encoding="utf-8"))
dataR = L.load_universe(L.UNIVERSE)
ctx = {"btc_ok": R1.btc_filter(dataR["BTCUSDT"])}
nr = 0
for name, (uni, a, b) in RC.SPLITS.items():
    for mode in ("as_configured", "no_halt"):
        for label, rules in (("R0", RC.PM.Rules()), ("X1", RC.X1)):
            res, risk, fr = RC.book(dataR, ctx, uni, a, b, mode == "no_halt", rules)
            pos = RC.positions(res.trades)
            prof = {**RC.risk_profile(res, risk, fr), "book_total_R": float(pos.r.sum()),
                    "book_mean_R": float(pos.r.mean()) if len(pos) else np.nan}
            compare(f"reconcile {name} {label}_{mode}", rc["splits"][name][f"{label}_{mode}"], prof)
            nr += 1
for cell in rc["checks"]["risk_variant_cells"]:
    uni = L.MAIN if cell["universe"] == "main9" else L.HOLDOUT
    rules = RC.PM.Rules() if cell["variant"] == "R0" else RC.X1
    res, _, _ = RC.book(dataR, ctx, uni, "2019-01-01", "2025-01-01", cell["mode"] == "no_halt", rules)
    s = stats(res, 10000.0)
    compare(f"reconcile cell {cell['universe']} {cell['mode']} {cell['variant']}",
            {"recomputed_return": cell["recomputed_return"], "recomputed_dd": cell["recomputed_dd"]},
            {"recomputed_return": s["total_return_pct"], "recomputed_dd": s["max_dd_pct"]})
    nr += 1
print(f"Reconciliation: {nr} profiles/cells compared ({time.time() - t0:.0f}s)")
print(f"{worst['fields']} numeric fields compared; largest absolute difference: {worst['diff']:.3g} "
      f"({worst['where']})")
print("MISMATCHES:" if mismatches else "no mismatches", *mismatches[:30], sep="\n  ")
sys.exit(1 if mismatches else 0)
