"""Round 2 historical checks, exactly as pre-registered in docs/research/ROUND2_PROGRAMME.md section B.

    python research/round2.py      -> docs/research/06_round2.md, docs/research/round2_results.json,
                                      model cards in docs/research/ml_models/ (H2-ML7 / H2-ML8)

RESEARCH ONLY. Data 2019-2024 (already examined in Round 1): these checks can reject ideas or show fragility,
they cannot by themselves establish an edge. No rule of frozen T50/RS3 or of X1/X2 is changed by anything here.
Env R2_N_RANDOM (default 300) random-skip draws; R2_N_GENERIC (default 100) random-entry books.
"""
import hashlib
import inspect
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import ml_lab as L  # noqa: E402
import ml_round1 as M  # noqa: E402
import round1 as R1  # noqa: E402
import portfolio_variants as PV  # noqa: E402
from portfolio import Costs, stats  # noqa: E402

OUT_MD = os.path.join(ROOT, "docs", "research", "06_round2.md")
OUT_JSON = os.path.join(ROOT, "docs", "research", "round2_results.json")
M.N_RANDOM = int(os.environ.get("R2_N_RANDOM", "300"))
N_GENERIC = int(os.environ.get("R2_N_GENERIC", "100"))
BANDS = [(0.12, 0.08), (0.15, 0.10), (0.20, 0.15)]
LOCKS = [(1.5, 5.0), (2.0, 6.0), (3.0, 8.0), None]
STRESS = Costs(0.004, 0.002, 0.005)
ML7_FEATURES = ["btc_ma200_dist", "btc_ret_60", "breadth_ma50", "btc_vol_20"]
ML8_FEATURES = ["atr_pct", "vol_expansion", "brk_atr", "exit_dist_atr", "btc_vol_20"]


def sparse_logit():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))


MIN_ROWS = 50


def fit_sparse(X, y):
    ok = X.notna().all(axis=1) & y.notna()
    if ok.sum() < MIN_ROWS or y[ok].nunique() < 2:
        raise ValueError(f"cannot train: {int(ok.sum())} rows, {y[ok].nunique()} classes")
    m = sparse_logit()
    m.fit(X[ok].to_numpy(float), y[ok].to_numpy(int))
    return m, int(ok.sum())


def proba(m, X):
    out = pd.Series(np.nan, index=X.index)
    ok = X.notna().all(axis=1)
    if ok.any():
        out[ok] = m.predict_proba(X[ok].to_numpy(float))[:, 1]
    return out


def card(name, year, features, target, n, dhash, path, train_note):
    src = (inspect.getsource(sys.modules[__name__]) + L.code_hash()).replace("\r\n", "\n")
    with open(path, "rb") as f:
        mh = hashlib.sha256(f.read()).hexdigest()
    c = {"model": name, "fold_test_year": year, "kind": "logit C=0.1 (StandardScaler + LogisticRegression)",
         "target": target, "features": features, "training": train_note, "n_train": n,
         "embargo_days": L.EMBARGO_DAYS, "code_sha256": hashlib.sha256(src.encode()).hexdigest(),
         "data_sha256": dhash, "model_file": os.path.relpath(path, ROOT).replace("\\", "/"), "model_sha256": mh}
    c["card_sha256"] = hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()
    return c


# ------------------------------------------------------------------ position management checks
def pm_cells(data, ctx, costs=Costs(), configs=None):
    import t50_scrutiny as S
    rows = []
    for label, uni in (("main9", L.MAIN), ("holdout16", L.HOLDOUT)):
        fr, en, ex, at = S.portfolio_inputs({s: data[s] for s in uni}, ctx)
        for mode in ("as_configured", "no_halt"):
            for name, kw in configs.items():
                res, _ = PV.simulate(fr, en, ex, at, *R1.ALL, M.conservative(mode == "no_halt"), costs,
                                     capital=10000.0, **kw)
                s = stats(res, 10000.0)
                rows.append({"universe": label, "mode": mode, "config": name, "return_pct": s["total_return_pct"],
                             "max_dd_pct": s["max_dd_pct"], "ret_over_dd": s["total_return_pct"] / max(s["max_dd_pct"], 1e-9),
                             "sharpe": s["sharpe"]})
    return pd.DataFrame(rows)


def neighbours():
    cfgs = {"R0": {}}
    for b in BANDS:
        for lk in LOCKS:
            cfgs[f"band{b[0]:.2f}/{b[1]:.2f}+" + ("lock-off" if lk is None else f"lock{lk[0]}R/{lk[1]}ATR")] = \
                {"trim_band": b, **({"profit_lock": lk} if lk else {})}
    return cfgs


def improves(df, cfg):
    out = []
    for (u, m), g in df.groupby(["universe", "mode"]):
        b = g[g.config == "R0"].iloc[0]
        x = g[g.config == cfg].iloc[0]
        out.append(bool(x.ret_over_dd > b.ret_over_dd))
    return out


def generic(data, ctx, log):
    """R0 vs R3 on random-entry books (main 9): per-coin entry probability = the coin's 2019-24 T50 entry rate."""
    import t50_scrutiny as S
    d = {s: data[s] for s in L.MAIN}
    fr, en, ex, at = S.portfolio_inputs(d, ctx)
    rates = {p: float(e.mean()) for p, e in en.items()}
    rng = np.random.default_rng(4404)
    rows = []
    for i in range(N_GENERIC):
        ren = {p: pd.Series(rng.random(len(e)) < rates[p], index=e.index) for p, e in en.items()}
        for mode in ("as_configured", "no_halt"):
            r = {}
            for name, kw in (("R0", {}), ("R3", {"trim_band": (0.15, 0.10), "profit_lock": (2.0, 6.0)})):
                res, _ = PV.simulate(fr, ren, ex, at, *R1.ALL, M.conservative(mode == "no_halt"), Costs(),
                                     capital=10000.0, **kw)
                s = stats(res, 10000.0)
                r[name] = s["total_return_pct"] / max(s["max_dd_pct"], 1e-9)
            rows.append({"draw": i, "mode": mode, "R0_ret_dd": r["R0"], "R3_ret_dd": r["R3"]})
    log(f"  generic random-entry books: {N_GENERIC} draws")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ ML7 / ML8 gates
def ml7_gate(panel, signal_days, dhash, cards, log):
    sd = pd.DataFrame(signal_days, columns=["pair", "day"])
    pn = panel.set_index(["pair", "day"])
    gate, evals = {}, []
    for year in L.TEST_YEARS:
        tr, te = L.fold_masks(panel["day"], panel["label_end"], year)
        m, n = fit_sparse(panel.loc[tr, ML7_FEATURES], panel.loc[tr, "y_up"])
        base = float(panel.loc[tr, "y_up"].mean())
        path = os.path.join(M.MODELS, f"ML7_{year}.pkl")
        os.makedirs(M.MODELS, exist_ok=True)
        joblib.dump(m, path)
        cards.append(card("ML7", year, ML7_FEATURES, "y_up", n, dhash, path, f"label known before {year}-01-01 - 5d"))
        test = panel[te]
        p = proba(m, test[ML7_FEATURES])
        ok = p.notna() & test["y_up"].notna()
        ll_m = L.log_loss_rows(test.loc[ok, "y_up"].to_numpy(), p[ok].to_numpy()).mean()
        ll_c = L.log_loss_rows(test.loc[ok, "y_up"].to_numpy(), np.full(ok.sum(), base)).mean()
        from sklearn.metrics import roc_auc_score
        evals.append({"model": "ML7", "test_year": year, "train_rows": n, "base_rate": base,
                      "oos_logloss": ll_m, "oos_logloss_clim": ll_c,
                      "oos_auc": float(roc_auc_score(test.loc[ok, "y_up"], p[ok]))})
        ys = sd[sd.day.dt.year == year]
        pv = proba(m, pn[ML7_FEATURES].reindex(pd.MultiIndex.from_frame(ys)).reset_index(drop=True))
        for (pair, day), v in zip(zip(ys.pair, ys.day), pv):
            gate.setdefault(pair, {})[day] = bool(not np.isfinite(v) or v >= base - 0.10)
    log("  ML7 trained")
    return {p: pd.Series(v) for p, v in gate.items()}, evals


def ml8_gate(tf, panel, signal_days, dhash, cards, log):
    tf = tf.copy()
    tf["y_fast"] = ((tf["exit_reason"] == "stop_loss") & (tf["bars_held"] <= 5)).astype(float)
    feats = panel.set_index(["pair", "day"])[ML8_FEATURES]
    gate, evals = {}, []
    from sklearn.metrics import roc_auc_score
    for year in L.TEST_YEARS:
        start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        tr = tf["label_end"] < start - pd.Timedelta(days=L.EMBARGO_DAYS)
        m, n = fit_sparse(tf.loc[tr, ML8_FEATURES], tf.loc[tr, "y_fast"])
        path = os.path.join(M.MODELS, f"ML8_{year}.pkl")
        joblib.dump(m, path)
        cards.append(card("ML8", year, ML8_FEATURES, "y_fast (stop-out within 5 bars)", n, dhash, path,
                          f"trades exited before {year}-01-01 - 5d"))
        q = float(proba(m, tf.loc[tr, ML8_FEATURES]).dropna().quantile(0.80))
        te = tf[tf["signal_day"].dt.year == year]
        pt = proba(m, te[ML8_FEATURES])
        ok = pt.notna()
        evals.append({"model": "ML8", "test_year": year, "train_trades": n,
                      "train_rate": float(tf.loc[tr, "y_fast"].mean()), "threshold_p80": q,
                      "oos_auc": float(roc_auc_score(te.loc[ok, "y_fast"], pt[ok])) if te.loc[ok, "y_fast"].nunique() == 2 else np.nan})
        sd = [(p, d) for p, d in signal_days if d.year == year]
        pv = proba(m, feats.reindex(pd.MultiIndex.from_tuples(sd, names=["pair", "day"])).reset_index(drop=True))
        for (pair, day), v in zip(sd, pv):
            gate.setdefault(pair, {})[day] = bool(not np.isfinite(v) or v <= q)
    log("  ML8 trained")
    return {p: pd.Series(v) for p, v in gate.items()}, evals


def main():
    t0 = time.time()
    logs = []

    def log(m):
        print(m, flush=True)
        logs.append(m)

    data = L.load_universe()
    dhash = L.data_hashes(data)
    ctx = {"btc_ok": R1.btc_filter(data["BTCUSDT"])}
    panel = L.build_panel(data)
    out, cards = {}, []

    # H2-PM-ROBUST
    nb = neighbours()
    rob = pm_cells(data, ctx, configs=nb)
    share = {c: improves(rob, c) for c in nb if c != "R0"}
    frac = float(np.mean([all(v) for v in share.values()]))
    out["H2-PM-ROBUST"] = {"verdict": "supported" if frac >= 0.75 else "rejected",
                           "share_of_neighbours_improving_all_4_cells": frac,
                           "per_neighbour_cells_improved": {c: int(sum(v)) for c, v in share.items()}}
    log(f"  robustness: {frac:.0%} of 12 neighbours improve ret/DD in all 4 cells")

    # H2-PM-COST
    cost = pm_cells(data, ctx, STRESS, {"R0": {}, "R3": {"trim_band": (0.15, 0.10), "profit_lock": (2.0, 6.0)}})
    ok = improves(cost, "R3")
    out["H2-PM-COST"] = {"verdict": "supported" if all(ok) else "rejected", "cells_improved": int(sum(ok))}

    # H2-PM-GENERIC
    gen = generic(data, ctx, log)
    out["H2-PM-GENERIC"] = {"verdict": "descriptive",
                            **{m: float((g.R3_ret_dd > g.R0_ret_dd).mean()) for m, g in gen.groupby("mode")}}

    # H2-ML7 / H2-ML8 through the Round 1 gate evaluation (unchanged rules)
    c = M.cfg()
    t_all = M.trades_of(M.run_t50(data, ctx, c, "2019-01-01", M.WF_END), "2019-01-01", M.WF_END)
    signal_days = []
    for pair, df in data.items():
        sig = R1.signals("T50", df, {**ctx, "pair": pair})
        ts = pd.to_datetime(df["timestamp"], utc=True)
        signal_days += [(pair, d) for d in ts[sig["entry"].to_numpy(bool)]]
    tf = M.trade_frame(t_all, panel)
    g7, e7 = ml7_gate(panel, signal_days, dhash, cards, log)
    g8, e8 = ml8_gate(tf, panel, signal_days, dhash, cards, log)
    base_t = M.trades_of(M.run_t50(data, ctx, c, M.WF_START, M.WF_END))
    base = {**M.summary(base_t), "_t": base_t}
    base_stress = M.summary(M.trades_of(M.run_t50(data, ctx, M.cfg(True), M.WF_START, M.WF_END)))
    base_port = {m: M.portfolio_run(data, ctx, None, M.WF_START, M.WF_END, m == "no_halt") for m in ("as_configured", "no_halt")}
    gres = {}
    for name, g in (("ML7", g7), ("ML8", g8)):
        gres[name], _ = M.evaluate_gate(name, g, data, ctx, base, base_stress, base_port, signal_days, log)
    holm = L.holm({n: gres[n]["p_random"] for n in gres})
    for name in gres:
        v, checks = M.gate_verdict(gres[name], holm[name])
        out[f"H2-{name}"] = {"verdict": v, "checks": checks, "holm_p": holm[name]}

    os.makedirs(M.CARDS, exist_ok=True)
    for cd in cards:
        with open(os.path.join(M.CARDS, f"{cd['model']}_{cd['fold_test_year']}.json"), "w", encoding="utf-8") as f:
            json.dump(cd, f, indent=1, default=str)
    res = {"generated_utc": pd.Timestamp.now(tz="UTC").isoformat(), "verdicts": out, "robustness": rob.to_dict("records"),
           "cost_stress": cost.to_dict("records"), "generic": gen.to_dict("records"), "ml_evals": e7 + e8,
           "gates": gres, "base_G0": {k: v for k, v in base.items() if k != "_t"}, "data_sha256": dhash,
           "runtime_s": time.time() - t0}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    rows = [{"gate": n, "blocked": f"{r['signal_days_blocked']}/{r['signal_days_total']}", "trades": r["g_trades"],
             "total_R": r["g_total_R"], "dTotal_R": r["delta_total_R"], "p_random": r["p_random"], "p_holm": holm[n],
             "years_up": r["years_improved"], "d_main9": r["delta_main9"], "d_holdout16": r["delta_holdout16"],
             "removed_mean_R": r["removed_mean_R"], "d_cost_stress": r["delta_total_R_cost_stress"]} for n, r in gres.items()]
    piv = rob.pivot_table(index="config", columns=["universe", "mode"], values="ret_over_dd")
    txt = ["# Round 2 historical checks (pre-registered in `ROUND2_PROGRAMME.md`)\n",
           "Data 2019–2024 only, **already examined in Round 1**: these results can reject or flag fragility; "
           f"they cannot establish an edge. Reproduce: `python research/round2.py` (runtime {res['runtime_s'] / 60:.0f} min).\n",
           "## Verdicts\n", "| Hypothesis | Verdict | Detail |", "|---|---|---|"]
    for k, v in out.items():
        txt.append(f"| {k} | **{v['verdict']}** | {json.dumps({a: b for a, b in v.items() if a != 'verdict'}, default=str)[:300]} |")
    txt += ["\n## Position management: return / max DD for 12 pre-declared neighbours (2019–24)\n", M.md(piv, idx=True),
            "\n## Cost stress (0.4% fee, 0.2% slippage, 0.5% stop slippage)\n", M.md(cost),
            "\n## Random-entry books (main 9): share of draws where R3 raises return/DD\n",
            M.md(gen.groupby("mode").apply(lambda g: pd.Series({"draws": len(g), "share_R3_better": (g.R3_ret_dd > g.R0_ret_dd).mean(),
                                                                "median_R0": g.R0_ret_dd.median(), "median_R3": g.R3_ret_dd.median()})), idx=True),
            "\n## Sparse ML gates on frozen T50 (2021–24, 25 coins)\n",
            f"G0: {base['trades']} trades, total {base['total_R']:+.1f}R.\n", M.md(pd.DataFrame(rows)),
            "\n**Model evaluation per fold:**\n", M.md(pd.DataFrame(e7)), "", M.md(pd.DataFrame(e8)),
            "\n## Run log\n", "```", *logs, "```"]
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(txt) + "\n")
    log(f"done in {time.time() - t0:.0f}s")
    return res


if __name__ == "__main__":
    main()
