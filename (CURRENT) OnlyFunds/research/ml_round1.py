"""ML Round 1: runs exactly the experiments pre-registered in docs/research/ML_PROGRAMME.md (commit c0854be).

    python research/ml_round1.py            -> docs/research/04_ml_round1.md, docs/research/ml_round1_results.json,
                                               model cards in docs/research/ml_models/, models in state/ml/models/

RESEARCH ONLY: no orders, no keys, no exchange client; data on or after 2025-01-01 is never loaded; the frozen
T50/RS3 shadow tracking is not touched. Nothing is tuned: every model, threshold and rule is the pre-registered one.
Env ML_N_RANDOM (default 300) sets the random-skip draws; ML_JOBS the worker processes.
"""
import copy
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
os.environ["PYTHONPATH"] = os.pathsep.join([ROOT, HERE, os.environ.get("PYTHONPATH", "")])   # for worker processes
import ml_lab as L  # noqa: E402
import round1 as R1  # noqa: E402
import portfolio_variants as PV  # noqa: E402
from portfolio import Costs, stats  # noqa: E402
from core import validation as V  # noqa: E402
from core.backtester import run_backtest  # noqa: E402
from core.risk_manager import RiskLimits  # noqa: E402

OUT_MD = os.path.join(ROOT, "docs", "research", "04_ml_round1.md")
OUT_JSON = os.path.join(ROOT, "docs", "research", "ml_round1_results.json")
CARDS = os.path.join(ROOT, "docs", "research", "ml_models")
MODELS = os.path.join(os.environ.get("OF2_STATE_DIR") or os.path.join(ROOT, "state"), "ml", "models")
N_RANDOM = int(os.environ.get("ML_N_RANDOM", "300"))
JOBS = int(os.environ.get("ML_JOBS", str(max(1, (os.cpu_count() or 2) - 2))))
WF_START, WF_END = "2021-01-01", "2025-01-01"
STRESS = dict(fee=0.004, slip=0.002, stop_slip=0.005)
ML_GATES, RULE_GATES = ["GM1", "GM2", "GM3", "GM4"], ["G1", "G2", "G3"]
SMOKE = os.environ.get("ML_SMOKE")              # a directory: crash-test on 5 coins, outputs there (not the report)
if SMOKE:
    OUT_MD, OUT_JSON, CARDS, MODELS = (os.path.join(SMOKE, x) for x in ("smoke.md", "smoke.json", "cards", "models"))
    L.MIN_TRAIN_ROWS = 10
HYP = {"GM1": "H-ML3", "GM2": "H-ML4", "GM3": "H-ML5", "GM4": "H-ML6", "G1": "H-G1", "G2": "H-G2", "G3": "H-G3"}


def conservative(no_halt=False):
    return RiskLimits(risk_per_trade=0.005, max_position_pct=0.10, max_exposure_pct=0.30, max_open_positions=3,
                      max_daily_loss_pct=1.0 if no_halt else 0.03, max_drawdown_pct=1.0 if no_halt else 0.10,
                      min_notional=1.0, max_cost_to_risk=0.25, kill_switch_file="")


def cfg(stress=False):
    c = R1.config()
    c.close_at_end = True
    if stress:
        c.fee_rate, c.slippage_pct, c.stop_slippage_pct = STRESS["fee"], STRESS["slip"], STRESS["stop_slip"]
    return c


def md(df, idx=False):
    d = df.reset_index() if idx else df.copy()
    for col in d.columns:
        if d[col].dtype.kind == "f":
            d[col] = d[col].map(lambda v: "nan" if pd.isna(v) else f"{v:+.3f}" if abs(v) < 10 else f"{v:+.1f}")
    return ("| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns) + "\n" +
            "\n".join("| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False)))


# ------------------------------------------------------------------ engine runs (frozen T50, optional gate)
def run_t50(data, ctx, c, start, end, gate=None):
    out = {}
    for pair, df in data.items():
        seg = V.segment(df, start, end)
        ts = pd.to_datetime(seg["timestamp"], utc=True)
        if seg.empty or (ts >= pd.Timestamp(start, tz="UTC")).sum() < 50:
            continue
        cc = copy.deepcopy(c)
        cc.strategy, cc.trade_from = "trend_following", start
        sig = R1.signals("T50", seg, {**ctx, "pair": pair})
        if gate is not None and pair in gate:
            sig["entry"] = sig["entry"].to_numpy(bool) & gate[pair].reindex(ts).fillna(True).to_numpy(bool)
        out[pair] = run_backtest(seg, cc, pair, signals=sig)
    return out


def trades_of(res, start=WF_START, end=WF_END):
    parts = [r.trades.assign(pair=p) for p, r in res.items() if len(r.trades)]
    if not parts:
        return pd.DataFrame(columns=["pair", "entry_time", "exit_time", "r_multiple", "pnl"])
    t = pd.concat(parts, ignore_index=True)
    t["entry_ts"] = pd.to_datetime(t["entry_time"], utc=True)
    t["exit_ts"] = pd.to_datetime(t["exit_time"], utc=True)
    t = t[(t.entry_ts >= pd.Timestamp(start, tz="UTC")) & (t.entry_ts < pd.Timestamp(end, tz="UTC"))]
    t["signal_day"] = t["entry_ts"].dt.normalize() - pd.Timedelta(days=1)
    t["year"] = t["entry_ts"].dt.year
    t["week"] = t["entry_ts"].dt.strftime("%G-%V")
    return t.reset_index(drop=True)


def summary(t):
    if len(t) == 0:
        return {"trades": 0, "total_R": 0.0, "net_R": np.nan, "pf": np.nan, "win_pct": np.nan, "ci_lo": np.nan,
                "ci_hi": np.nan, "net_R_ex_top10pct": np.nan}
    gl = -t.loc[t.pnl <= 0, "pnl"].sum()
    lo, hi = L.week_cluster_ci(t.r_multiple.to_numpy(), t.week.to_numpy())
    k = int(np.ceil(len(t) * 0.10))
    rest = t.r_multiple.sort_values().iloc[:len(t) - k]
    return {"trades": int(len(t)), "total_R": float(t.r_multiple.sum()), "net_R": float(t.r_multiple.mean()),
            "pf": float(t.loc[t.pnl > 0, "pnl"].sum() / gl) if gl > 0 else np.inf,
            "win_pct": float((t.pnl > 0).mean() * 100), "ci_lo": lo, "ci_hi": hi,
            "net_R_ex_top10pct": float(rest.mean()) if len(rest) else np.nan}


def _random_batch(data, ctx, c, signal_days, k, seeds, start, end):
    return [_random_draw(data, ctx, c, signal_days, k, s, start, end) for s in seeds]


def _random_draw(data, ctx, c, signal_days, k, seed, start, end):
    rng = np.random.default_rng(seed)
    pick = set(rng.choice(len(signal_days), size=k, replace=False).tolist()) if k > 0 else set()
    gate = {}
    for i, (pair, day) in enumerate(signal_days):
        if i in pick:
            gate.setdefault(pair, {})[day] = False
    gate = {p: pd.Series(v) for p, v in gate.items()}
    return float(trades_of(run_t50(data, ctx, c, start, end, gate), start, end).r_multiple.sum())


# ------------------------------------------------------------------ coin-day models
def coin_day_models(panel, dhash, log):
    preds, cards, fits = [], [], {}
    for year in L.TEST_YEARS:
        tr, te = L.fold_masks(panel["day"], panel["label_end"], year)
        test = panel[te].copy()
        for name in ("M1", "M2", "M1s", "M2s", "MV"):
            kind, target = L.MODEL_SPECS[name]
            m, n = L.fit(kind, panel.loc[tr, L.FEATURES], panel.loc[tr, target])
            path = os.path.join(MODELS, f"{name}_{year}.pkl")
            os.makedirs(MODELS, exist_ok=True)
            joblib.dump(m, path)
            first = str(panel.loc[tr & panel[L.FEATURES].notna().all(axis=1), "day"].min().date())
            cards.append(L.model_card(name, year, kind, target, L.FEATURES, (first, f"{year}-01-01 minus {L.EMBARGO_DAYS}d"),
                                      n, dhash, path))
            test[f"p_{name}"] = L.predict(kind, m, test[L.FEATURES])
            trp = L.predict(kind, m, panel.loc[tr, L.FEATURES])
            ytr = panel.loc[tr, target]
            ok = trp.notna() & ytr.notna()
            if kind != "ridge" and ok.any():
                fits[(name, year)] = float(roc_auc_score(ytr[ok], trp[ok]))
        for target in ("y_up", "y_stress"):
            base = panel.loc[tr, target].mean()
            test[f"clim_{target}"] = base
            by = panel.loc[tr].groupby("btc_class")[target].mean()
            test[f"rule_{target}"] = test["btc_class"].map(by).fillna(base).astype(float)
        test["clim_y_vol"] = panel.loc[tr, "y_vol"].mean()
        preds.append(test)
        log(f"  coin-day models fold {year}: train rows {int(tr.sum())}, test rows {int(te.sum())}")
    return pd.concat(preds, ignore_index=True), cards, fits


def prob_eval(oos, name, target):
    rows, out = [], {}
    d = oos[oos[target].notna() & oos[f"p_{name}"].notna()].copy()
    d["month"] = d["day"].dt.strftime("%Y-%m")
    y = d[target].to_numpy()
    for span, mask in (("2021-24", np.ones(len(d), bool)), ("DEV-OOS 2021-22", d["day"].dt.year.isin([2021, 2022]).to_numpy()),
                       ("VAL-OOS 2023-24", d["day"].dt.year.isin([2023, 2024]).to_numpy())):
        s = d[mask]
        ys = y[mask]
        ll_m = L.log_loss_rows(ys, s[f"p_{name}"].to_numpy())
        res = {"rows": int(mask.sum()), "base_rate": float(ys.mean()), "logloss_model": float(ll_m.mean())}
        for b in ("clim", "rule"):
            ll_b = L.log_loss_rows(ys, s[f"{b}_{target}"].to_numpy())
            m, lo, hi, p = L.block_bootstrap_mean(ll_b - ll_m, s["month"].to_numpy())
            res.update({f"logloss_{b}": float(ll_b.mean()), f"gain_vs_{b}": m, f"gain_vs_{b}_lo": lo,
                        f"gain_vs_{b}_hi": hi, f"p_vs_{b}": p})
        res["auc"] = float(roc_auc_score(ys, s[f"p_{name}"])) if len(np.unique(ys)) == 2 else np.nan
        res["auc_rule"] = float(roc_auc_score(ys, s[f"rule_{target}"])) if len(np.unique(ys)) == 2 else np.nan
        res["brier_model"] = float(np.mean((s[f"p_{name}"] - ys) ** 2))
        res["brier_clim"] = float(np.mean((s[f"clim_{target}"] - ys) ** 2))
        out[span] = res
        rows.append({"model": name, "target": target, "span": span, **res})
    rel = d.assign(bin=pd.qcut(d[f"p_{name}"], 10, labels=False, duplicates="drop")).groupby("bin").agg(
        predicted=(f"p_{name}", "mean"), observed=(target, "mean"), rows=(target, "size"))
    return out, rows, rel


def vol_eval(oos):
    d = oos[oos["y_vol"].notna() & oos["p_MV"].notna() & oos["vol_persist"].notna()].copy()
    out = {}
    for span, mask in (("2021-24", np.ones(len(d), bool)), ("DEV-OOS 2021-22", d["day"].dt.year.isin([2021, 2022]).to_numpy()),
                       ("VAL-OOS 2023-24", d["day"].dt.year.isin([2023, 2024]).to_numpy())):
        s = d[mask]
        e_m = (s["y_vol"] - s["p_MV"]) ** 2
        e_p = (s["y_vol"] - s["vol_persist"]) ** 2
        e_c = (s["y_vol"] - s["clim_y_vol"]) ** 2
        m, lo, hi, p = L.block_bootstrap_mean((e_p - e_m).to_numpy(), s["day"].dt.strftime("%Y-%m").to_numpy())
        out[span] = {"rows": int(len(s)), "skill_vs_persistence": float(1 - e_m.mean() / e_p.mean()),
                     "skill_vs_climatology": float(1 - e_m.mean() / e_c.mean()), "mse_gain_lo": lo, "mse_gain_hi": hi}
    return out


# ------------------------------------------------------------------ trade models
def trade_frame(t_all, panel):
    """Frozen-T50 trades (2019-24) with decision-day features and the coin's prior-trade record (causal)."""
    feats = panel.set_index(["pair", "day"])[L.FEATURES]
    t = t_all.copy()
    t = t.join(feats, on=["pair", "signal_day"])
    t["y_win"] = (t["r_multiple"] > 0).astype(float)
    t["label_end"] = t["exit_ts"]
    t["prior_r_shrunk"] = [prior_r(t_all, p, d) for p, d in zip(t["pair"], t["signal_day"])]
    return t


def prior_r(t_all, pair, day):
    prev = t_all[(t_all.pair == pair) & (t_all.exit_ts <= day)]
    return float(prev.r_multiple.sum() / (len(prev) + 5))


def trade_models(tf, panel, t_all, signal_days, dhash, log):
    """Per fold: train M3/M4 on trades known before the test year; score every T50 entry signal in the year."""
    feats = panel.set_index(["pair", "day"])[L.FEATURES]
    scores, cards, evals, fits = {}, [], [], {}
    for year in L.TEST_YEARS:
        start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        tr = tf["label_end"] < start - pd.Timedelta(days=L.EMBARGO_DAYS)
        sd = [(p, d) for p, d in signal_days if d.year == year]
        X = feats.reindex(pd.MultiIndex.from_tuples(sd, names=["pair", "day"])).reset_index()
        X["prior_r_shrunk"] = [prior_r(t_all, p, d) for p, d in sd]
        te_trades = tf[tf["signal_day"].dt.year == year]
        for name in ("M3", "M4"):
            kind, target = L.MODEL_SPECS[name]
            m, n = L.fit(kind, tf.loc[tr, L.TRADE_FEATURES], tf.loc[tr, target])
            path = os.path.join(MODELS, f"{name}_{year}.pkl")
            joblib.dump(m, path)
            cards.append(L.model_card(name, year, kind, target, L.TRADE_FEATURES,
                                      (str(tf.loc[tr, "signal_day"].min().date()), f"exit before {year}-01-01 minus {L.EMBARGO_DAYS}d"),
                                      n, dhash, path))
            ptr = L.predict(kind, m, tf.loc[tr, L.TRADE_FEATURES])
            q = float(ptr.dropna().quantile(1 / 3))
            p = L.predict(kind, m, X[L.TRADE_FEATURES])
            for (pair, day), pv in zip(sd, p):
                scores.setdefault(name, {}).setdefault(pair, {})[day] = bool(not np.isfinite(pv) or pv >= q)
            pt = L.predict(kind, m, te_trades[L.TRADE_FEATURES])
            ok = pt.notna()
            yt = te_trades.loc[ok, "y_win"]
            ok_tr = ptr.notna()
            fits[(name, year)] = float(roc_auc_score(tf.loc[tr, target][ok_tr], ptr[ok_tr])) \
                if tf.loc[tr, target][ok_tr].nunique() == 2 else np.nan
            evals.append({"model": name, "test_year": year, "train_trades": n, "test_trades": int(ok.sum()),
                          "threshold_q33": q, "auc_oos": float(roc_auc_score(yt, pt[ok])) if yt.nunique() == 2 else np.nan,
                          "auc_train": fits[(name, year)]})
        log(f"  trade models fold {year}: train trades {int(tr.sum())}, signals scored {len(sd)}")
    gates = {n: {p: pd.Series(v) for p, v in d.items()} for n, d in scores.items()}
    return gates, cards, pd.DataFrame(evals)


# ------------------------------------------------------------------ gates
def build_gates(panel, oos, data, ctx, signal_days, trade_gates):
    sd = pd.DataFrame(signal_days, columns=["pair", "day"])
    sd = sd[(sd.day >= pd.Timestamp(WF_START, tz="UTC")) & (sd.day < pd.Timestamp(WF_END, tz="UTC"))]
    o = oos.set_index(["pair", "day"])
    pn = panel.set_index(["pair", "day"])
    gates = {}
    for g, col in (("GM1", "p_M1"), ("GM2", "p_M2")):
        v = o[col].reindex(pd.MultiIndex.from_frame(sd))
        ok = (~np.isfinite(v.to_numpy())) | (v.to_numpy() >= 0.5)
        gates[g] = _to_gate(sd, ok)
    gates["GM3"], gates["GM4"] = trade_gates["M3"], trade_gates["M4"]
    btc_ok = ctx["btc_ok"]
    gates["G1"] = _to_gate(sd, btc_ok.reindex(sd.day).fillna(True).astype(bool).to_numpy())
    ma = pn["ma200_dist"].reindex(pd.MultiIndex.from_frame(sd)).to_numpy()
    gates["G2"] = _to_gate(sd, ~(ma <= 0))
    cls = pn["btc_class"].reindex(pd.MultiIndex.from_frame(sd)).to_numpy()
    gates["G3"] = _to_gate(sd, ~np.isin(cls.astype(str), ["weak_down", "strong_down"]))
    return gates


def _to_gate(sd, ok):
    gate = {}
    for (pair, day), v in zip(zip(sd.pair, sd.day), ok):
        gate.setdefault(pair, {})[day] = bool(v)
    return {p: pd.Series(v) for p, v in gate.items()}


def blocked_count(gate):
    return int(sum((~s).sum() for s in gate.values()))


def portfolio_run(data, ctx, gate, start, end, no_halt):
    import t50_scrutiny as S
    fr, en, ex, at = S.portfolio_inputs(data, ctx)
    if gate is not None:
        for p in en:
            if p in gate:
                en[p] = en[p] & gate[p].reindex(en[p].index).fillna(True).astype(bool)
    res, risk = PV.simulate(fr, en, ex, at, start, end, conservative(no_halt), Costs(), capital=10000.0)
    s = stats(res, 10000.0)
    s["cash_days_pct"] = float((res.n_open == 0).mean() * 100)
    s["ret_over_dd"] = s["total_return_pct"] / max(s["max_dd_pct"], 1e-9)
    return s


def evaluate_gate(name, gate, data, ctx, base, base_stress, base_port, signal_days, log):
    c = cfg()
    t = trades_of(run_t50(data, ctx, c, WF_START, WF_END, gate))
    ts = trades_of(run_t50(data, ctx, cfg(True), WF_START, WF_END, gate))
    k = blocked_count(gate)
    sdays = [(p, d) for p, d in signal_days if pd.Timestamp(WF_START, tz="UTC") <= d < pd.Timestamp(WF_END, tz="UTC")]
    t0 = time.time()
    batches = np.array_split(np.arange(50_000, 50_000 + N_RANDOM), JOBS)
    rand = [x for part in joblib.Parallel(n_jobs=JOBS)(
        joblib.delayed(_random_batch)(data, ctx, c, sdays, k, b.tolist(), WF_START, WF_END) for b in batches if len(b))
        for x in part]
    rand_delta = np.array(rand) - base["total_R"]
    delta = float(t.r_multiple.sum() - base["total_R"])
    by_year = {}
    t0y = base["_t"]
    for y in L.TEST_YEARS:
        by_year[y] = float(t[t.year == y].r_multiple.sum() - t0y[t0y.year == y].r_multiple.sum())
    sub = {}
    for label, uni in (("main9", L.MAIN), ("holdout16", L.HOLDOUT)):
        sub[label] = float(t[t.pair.isin(uni)].r_multiple.sum() - t0y[t0y.pair.isin(uni)].r_multiple.sum())
    # trades the gate removed (frozen-T50 trades whose decision day the gate rejects)
    rej = t0y[[not bool(gate.get(p, pd.Series(dtype=bool)).get(d, True)) for p, d in zip(t0y.pair, t0y.signal_day)]]
    port = {m: portfolio_run(data, ctx, gate, WF_START, WF_END, m == "no_halt") for m in ("as_configured", "no_halt")}
    out = {"gate": name, "signal_days_blocked": k, "signal_days_total": len(sdays), **{f"g_{a}": b for a, b in summary(t).items()},
           "delta_total_R": delta, "random_delta_mean": float(rand_delta.mean()), "random_delta_p95": float(np.percentile(rand_delta, 95)),
           "p_random": float(((rand_delta >= delta).sum() + 1) / (len(rand_delta) + 1)),
           "delta_by_year": by_year, "years_improved": int(sum(v > 0 for v in by_year.values())),
           "delta_main9": sub["main9"], "delta_holdout16": sub["holdout16"],
           "removed_trades": int(len(rej)), "removed_total_R": float(rej.r_multiple.sum()),
           "removed_mean_R": float(rej.r_multiple.mean()) if len(rej) else np.nan,
           "removed_ci": L.week_cluster_ci(rej.r_multiple.to_numpy(), rej.week.to_numpy()) if len(rej) > 1 else (np.nan, np.nan),
           "removed_top5_R": float(rej.r_multiple.nlargest(5).sum()) if len(rej) else 0.0,
           "delta_total_R_cost_stress": float(ts.r_multiple.sum() - base_stress["total_R"]),
           "portfolio": port,
           "ret_over_dd_vs_base": {m: port[m]["ret_over_dd"] - base_port[m]["ret_over_dd"] for m in port}}
    log(f"  gate {name}: blocked {k}/{len(sdays)} signal days, dTotalR {delta:+.2f}, random p {out['p_random']:.3f} "
        f"({time.time() - t0:.0f}s random)")
    return out, t


def gate_verdict(r, p_holm):
    checks = {
        "total_R_up": r["delta_total_R"] > 0,
        "beats_random_holm": p_holm < 0.05,
        "years_3_of_4": r["years_improved"] >= 3,
        "same_sign_main_holdout": r["delta_main9"] > 0 and r["delta_holdout16"] > 0,
        "portfolio_ret_dd_not_worse": all(v >= 0 for v in r["ret_over_dd_vs_base"].values()),
        "survives_cost_stress": r["delta_total_R_cost_stress"] > 0,
    }
    if all(checks.values()):
        v = "supported"
    elif r["delta_total_R"] > 0:
        v = "inconclusive"
    else:
        v = "rejected"
    return v, checks


# ------------------------------------------------------------------ capital preservation (P5)
VARIANTS = {"R0": {}, "R1": {"trim_band": (0.15, 0.10)}, "R2": {"profit_lock": (2.0, 6.0)},
            "R3": {"trim_band": (0.15, 0.10), "profit_lock": (2.0, 6.0)}}


def risk_variants(data, ctx, log):
    import t50_scrutiny as S
    rows, risk_base, results = [], {}, {}
    for label, uni in (("main9", L.MAIN), ("holdout16", L.HOLDOUT)):
        d = {s: data[s] for s in uni if s in data}
        fr, en, ex, at = S.portfolio_inputs(d, ctx)
        for mode in ("as_configured", "no_halt"):
            base_tr = None
            for v, kw in VARIANTS.items():
                res, risk = PV.simulate(fr, en, ex, at, "2019-01-01", "2025-01-01", conservative(mode == "no_halt"),
                                        Costs(), capital=10000.0, **kw)
                s = stats(res, 10000.0)
                tr = res.trades
                pos_pnl = tr.groupby(["pair", "entry_day"])["pnl"].sum() if len(tr) else pd.Series(dtype=float)
                if v == "R0":
                    base_tr = pos_pnl
                    risk_base[(label, mode)] = risk
                top5 = base_tr.nlargest(5)
                kept = float(pos_pnl.reindex(top5.index).fillna(0).sum() / top5.sum()) if top5.sum() > 0 else np.nan
                row = {"universe": label, "mode": mode, "variant": v, "return_pct": s["total_return_pct"],
                       "cagr_pct": s["cagr_pct"], "max_dd_pct": s["max_dd_pct"], "sharpe": s["sharpe"],
                       "ret_over_dd": s["total_return_pct"] / max(s["max_dd_pct"], 1e-9), "positions": int(len(pos_pnl)),
                       "trims": int((tr.reason == "trim").sum()) if len(tr) else 0,
                       "profit_lock_exits": int((tr.reason == "profit_lock").sum()) if len(tr) else 0,
                       "max_position_pct": float(risk["max_position_pct"].max()),
                       "max_open_risk_pct": float(risk["open_risk_to_stop_pct"].max()),
                       "top5_R0_profit_kept": kept, "halted_on": s["halted_on"]}
                rows.append(row)
        log(f"  risk variants {label} done")
    df = pd.DataFrame(rows)
    verdicts = {}
    for v in ("R1", "R2", "R3"):
        checks = []
        for (u, m), g in df.groupby(["universe", "mode"]):
            b, x = g[g.variant == "R0"].iloc[0], g[g.variant == v].iloc[0]
            checks.append({"universe": u, "mode": m, "dd_down_2pts": x.max_dd_pct <= b.max_dd_pct - 2,
                           "ret_dd_not_lower": x.ret_over_dd >= b.ret_over_dd,
                           "return_ge_80pct": x.return_pct >= 0.8 * b.return_pct if b.return_pct > 0 else x.return_pct >= b.return_pct,
                           "top5_kept_70pct": bool(x.top5_R0_profit_kept >= 0.7)})
        ok = all(all(v2 for k2, v2 in c.items() if k2 not in ("universe", "mode")) for c in checks)
        verdicts[v] = ("supported" if ok else "rejected", checks)
    return df, verdicts, risk_base


# ------------------------------------------------------------------ anomaly checks (AI research-assistant layer)
def anomaly_checks(prob, fits_cd, trade_eval, gate_res):
    flags = []
    for name in ("M1", "M2", "M1s", "M2s"):
        tr = [v for (n, y), v in fits_cd.items() if n == name]
        oos = prob[name]["2021-24"]["auc"]
        if tr and np.mean(tr) - oos > 0.10:
            flags.append(f"{name}: mean train AUC {np.mean(tr):.2f} vs OOS {oos:.2f} -> overfitting")
    for _, r in trade_eval.iterrows():
        if np.isfinite(r.auc_train) and np.isfinite(r.auc_oos) and r.auc_train - r.auc_oos > 0.10:
            flags.append(f"{r.model} {r.test_year}: train AUC {r.auc_train:.2f} vs OOS {r.auc_oos:.2f} -> overfitting")
    for name, spans in prob.items():
        a = spans["2021-24"]["auc"]
        if a > 0.70:
            flags.append(f"{name}: OOS AUC {a:.2f} is unusually high for a 20-day crypto return target -> check for leakage")
    for g, r in gate_res.items():
        p = r["portfolio"]["no_halt"]
        if p.get("sharpe", 0) > 3 or (np.isfinite(r["g_pf"]) and r["g_pf"] > 5):
            flags.append(f"{g}: suspiciously strong result (Sharpe {p.get('sharpe'):.2f}, PF {r['g_pf']:.2f})")
        if r["signal_days_blocked"] / max(r["signal_days_total"], 1) > 0.8:
            flags.append(f"{g}: blocks {r['signal_days_blocked']}/{r['signal_days_total']} signal days -> mostly a cash rule")
    return flags


# ------------------------------------------------------------------ main
def main():
    t_start = time.time()
    logs = []

    def log(m):
        print(m, flush=True)
        logs.append(m)

    data = L.load_universe(["BTCUSDT", "ETHUSDT", "BNBUSDT", "ETCUSDT", "LINKUSDT"] if SMOKE else L.UNIVERSE)
    dhash = L.data_hashes(data)
    ctx = {"btc_ok": R1.btc_filter(data["BTCUSDT"])}
    log(f"data: {len(data)} coins, last candle {max(pd.to_datetime(d['timestamp'], utc=True).max() for d in data.values()).date()}")
    panel = L.build_panel(data)
    log(f"panel: {len(panel)} coin-days")

    oos, cards, fits_cd = coin_day_models(panel, dhash, log)
    prob, prob_rows, rel = {}, [], {}
    for name, target in (("M1", "y_up"), ("M2", "y_up"), ("M1s", "y_stress"), ("M2s", "y_stress")):
        prob[name], rows, rel[name] = prob_eval(oos, name, target)
        prob_rows += rows
    vol = vol_eval(oos)

    # frozen T50 over 2019-24 (trade-model training data) and its entry-signal days
    c = cfg()
    t_all = trades_of(run_t50(data, ctx, c, "2019-01-01", WF_END), "2019-01-01", WF_END)
    signal_days = []
    for pair, df in data.items():
        sig = R1.signals("T50", df, {**ctx, "pair": pair})
        ts = pd.to_datetime(df["timestamp"], utc=True)
        signal_days += [(pair, d) for d in ts[sig["entry"].to_numpy(bool)]]
    tf = trade_frame(t_all, panel)
    trade_gates, tcards, trade_eval = trade_models(tf, panel, t_all, signal_days, dhash, log)
    cards += tcards

    base_t = trades_of(run_t50(data, ctx, c, WF_START, WF_END))
    base = {**summary(base_t), "_t": base_t}
    base_stress = summary(trades_of(run_t50(data, ctx, cfg(True), WF_START, WF_END)))
    base_port = {m: portfolio_run(data, ctx, None, WF_START, WF_END, m == "no_halt") for m in ("as_configured", "no_halt")}
    log(f"G0 frozen T50 2021-24: {base['trades']} trades, total {base['total_R']:+.1f}R, mean {base['net_R']:+.3f}R")

    gates = build_gates(panel, oos, data, ctx, signal_days, trade_gates)
    gate_res, gate_trades = {}, {}
    for g in ML_GATES + RULE_GATES:
        gate_res[g], gate_trades[g] = evaluate_gate(g, gates[g], data, ctx, base, base_stress, base_port, signal_days, log)
    # Family B: the three rule gates. Family A: H-ML1/H-ML2 (probabilistic skill) + the four ML gates (6 tests).
    p_holm = L.holm({g: gate_res[g]["p_random"] for g in RULE_GATES})
    p_ml_models = {n: max(prob[n]["2021-24"]["p_vs_clim"], prob[n]["2021-24"]["p_vs_rule"]) for n in ("M1", "M2")}
    famA = L.holm({**{f"H-{n}": p for n, p in p_ml_models.items()}, **{g: gate_res[g]["p_random"] for g in ML_GATES}})
    for g in ML_GATES:
        p_holm[g] = famA[g]
    verdicts = {}
    for n, hid in (("M1", "H-ML1"), ("M2", "H-ML2")):
        spans = prob[n]
        ok = famA[f"H-{n}"] < 0.05 and all(spans[s]["gain_vs_clim_lo"] > 0 and spans[s]["gain_vs_rule_lo"] > 0 for s in spans)
        any_gain = spans["2021-24"]["gain_vs_clim"] > 0 and spans["2021-24"]["gain_vs_rule"] > 0
        verdicts[hid] = ("supported" if ok else "inconclusive" if any_gain else "rejected",
                         {"holm_p": famA[f"H-{n}"], **{s: (round(v["gain_vs_clim_lo"], 5), round(v["gain_vs_rule_lo"], 5)) for s, v in spans.items()}})
    for g in ML_GATES + RULE_GATES:
        verdicts[HYP[g]] = gate_verdict(gate_res[g], p_holm[g])

    # RS3 with the GM1 gate (descriptive selection check)
    rs = {}
    main = {s: data[s] for s in L.MAIN if s in data}
    member, _ = R1.rs_basket(main, ctx["btc_ok"])
    rctx = {**ctx, "member": member}
    for label, gate in (("RS3", None), ("RS3+GM1", gates["GM1"])):
        out = {}
        for pair, df in main.items():
            seg = V.segment(df, WF_START, WF_END)
            ts = pd.to_datetime(seg["timestamp"], utc=True)
            cc = copy.deepcopy(c)
            cc.strategy, cc.trade_from = "trend_following", WF_START
            sig = R1.signals("RS3", seg, {**rctx, "pair": pair})
            if gate is not None:
                p_up = oos.set_index(["pair", "day"])["p_M1"]
                pv = p_up.reindex(pd.MultiIndex.from_arrays([[pair] * len(ts), ts])).to_numpy()
                sig["entry"] = sig["entry"].to_numpy(bool) & (~np.isfinite(pv) | (pv >= 0.5))
            out[pair] = run_backtest(seg, cc, pair, signals=sig)
        rs[label] = summary(trades_of(out))

    rv, rv_verdicts, risk_base = risk_variants(data, ctx, log)
    for v, (verdict, checks) in rv_verdicts.items():
        verdicts[f"H-{v}"] = (verdict, checks)
    flags = anomaly_checks(prob, fits_cd, trade_eval, gate_res)

    # cards
    os.makedirs(CARDS, exist_ok=True)
    for cd in cards:
        with open(os.path.join(CARDS, f"{cd['model']}_{cd['fold_test_year']}.json"), "w", encoding="utf-8") as f:
            json.dump(cd, f, indent=1, default=str)

    results = {"generated_utc": pd.Timestamp.now(tz="UTC").isoformat(), "preregistration_commit": "c0854be",
               "data_sha256": dhash, "n_random": N_RANDOM, "probabilistic": prob, "volatility": vol,
               "trade_model_eval": trade_eval.to_dict("records"),
               "base_G0": {k: v for k, v in base.items() if k != "_t"}, "base_G0_cost_stress": base_stress,
               "base_G0_portfolio": base_port, "gates": gate_res, "holm_p": p_holm, "familyA_holm": famA,
               "rs3_selection": rs, "risk_variants": rv.to_dict("records"),
               "verdicts": {k: {"verdict": v[0], "checks": v[1]} for k, v in verdicts.items()},
               "anomaly_flags": flags, "runtime_s": time.time() - t_start}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    write_report(results, prob_rows, rel, trade_eval, rv, risk_base, logs)
    log(f"done in {time.time() - t_start:.0f}s")
    return results


def write_report(R, prob_rows, rel, trade_eval, rv, risk_base, logs):
    g0 = R["base_G0"]
    L_ = ["# ML Round 1 results (pre-registered in `ML_PROGRAMME.md`, commit c0854be)\n",
          "Research only. Data 2019-01-01 → 2024-12-31 (25 coins, daily); nothing ≥ 2025 loaded; no tuning; the frozen "
          "T50/RS3 shadow tracking was not touched. Reproduce: `python research/ml_round1.py` "
          f"(random-skip draws: {R['n_random']}; runtime {R['runtime_s'] / 60:.0f} min).\n",
          "## Verdicts\n", "| Hypothesis | Verdict |\n|---|---|"]
    for k, v in R["verdicts"].items():
        L_.append(f"| {k} | **{v['verdict']}** |")
    L_ += ["\n## A. Market-state models: probabilistic skill out of sample (walk-forward 2021–24)\n",
           "Gain = mean log-loss improvement per coin-day (positive = model better); 95% interval from a month-block "
           "bootstrap (all coins in a month resampled together).\n"]
    pr = pd.DataFrame(prob_rows)[["model", "target", "span", "rows", "base_rate", "logloss_model", "gain_vs_clim",
                                    "gain_vs_clim_lo", "gain_vs_rule", "gain_vs_rule_lo", "auc", "auc_rule"]]
    L_.append(md(pr))
    L_.append("\n**Reliability (M1, OOS 2021–24, deciles of predicted probability):**\n")
    L_.append(md(rel["M1"].reset_index()))
    L_.append("\n**Volatility forecast (MV, log 20-day realised volatility):**\n")
    L_.append(md(pd.DataFrame(R["volatility"]).T.reset_index().rename(columns={"index": "span"})))
    L_ += ["\n## B. T50 trade-scoring models (M3 logistic, M4 boosting)\n", md(trade_eval)]
    L_ += ["\n## C/D. Gates on the frozen T50 entries (2021–24, 25 coins, unchanged engine)\n",
           f"G0 frozen T50: {g0['trades']} trades, total {g0['total_R']:+.1f}R, mean {g0['net_R']:+.3f}R "
           f"(95% {g0['ci_lo']:+.3f} to {g0['ci_hi']:+.3f}), PF {g0['pf']:.2f}, without top 10% {g0['net_R_ex_top10pct']:+.3f}R.\n"]
    rows = []
    for g, r in R["gates"].items():
        rows.append({"gate": g, "blocked_signal_days": f"{r['signal_days_blocked']}/{r['signal_days_total']}",
                     "trades": r["g_trades"], "total_R": r["g_total_R"], "mean_R": r["g_net_R"], "pf": r["g_pf"],
                     "dTotal_R": r["delta_total_R"], "random_skip_mean_d": r["random_delta_mean"],
                     "random_p95_d": r["random_delta_p95"], "p_random": r["p_random"], "p_holm": R["holm_p"][g],
                     "years_up": r["years_improved"], "d_main9": r["delta_main9"], "d_holdout16": r["delta_holdout16"],
                     "d_cost_stress": r["delta_total_R_cost_stress"]})
    L_.append(md(pd.DataFrame(rows)))
    L_.append("\n**Trades each gate removed** (frozen-T50 trades whose decision day the gate rejects; a gate that "
              "helps should remove trades with negative mean R):\n")
    rows = [{"gate": g, "removed": r["removed_trades"], "removed_total_R": r["removed_total_R"],
             "removed_mean_R": r["removed_mean_R"], "ci_lo": r["removed_ci"][0], "ci_hi": r["removed_ci"][1],
             "removed_top5_R": r["removed_top5_R"]} for g, r in R["gates"].items()]
    L_.append(md(pd.DataFrame(rows)))
    L_.append("\n**Single account, Conservative limits, 25 coins, 2021–24:**\n")
    rows = []
    for g, port in [("G0", R["base_G0_portfolio"])] + [(g, r["portfolio"]) for g, r in R["gates"].items()]:
        for m, s in port.items():
            rows.append({"gate": g, "mode": m, "return_pct": s["total_return_pct"], "max_dd_pct": s["max_dd_pct"],
                         "ret_over_dd": s["ret_over_dd"], "sharpe": s["sharpe"], "trades": s["trades"],
                         "cash_days_pct": s["cash_days_pct"], "halted_on": s["halted_on"]})
    L_.append(md(pd.DataFrame(rows)))
    L_.append("\n**Per-year change in total R:**\n")
    L_.append(md(pd.DataFrame({g: r["delta_by_year"] for g, r in R["gates"].items()}).T, idx=True))
    L_.append("\n**RS3 with the GM1 market-state gate (descriptive selection check, main 9, 2021–24):**\n")
    L_.append(md(pd.DataFrame(R["rs3_selection"]).T, idx=True))
    L_ += ["\n## P5. Capital preservation variants (single account, 2019–24)\n", md(rv)]
    rb = risk_base[("main9", "as_configured")]
    L_ += ["\n**Risk decomposition, R0 main 9 as configured (percent of equity):**\n",
           md(rb.drop(columns=["equity"]).describe().T[["mean", "50%", "max"]], idx=True)]
    L_ += ["\n## Automated anomaly checks\n"] + ([f"- {f}" for f in R["anomaly_flags"]] or ["- none raised"])
    L_ += ["\n## Run log\n", "```", *logs, "```"]
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L_) + "\n")


if __name__ == "__main__":
    main()
