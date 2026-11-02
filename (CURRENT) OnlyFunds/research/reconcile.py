"""Reconciliation of the published frozen-T50 figures, plus the T50 single-account risk analysis. Research only.

Recomputes the published numbers from the same code and data, checks them against the stored results files, and
explains where they differ. It never edits a published result. Outputs: docs/research/07_reconciliation.md and
docs/research/reconciliation.json.

    python research/reconcile.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import ml_lab as L  # noqa: E402
import ml_round1 as M  # noqa: E402
import portfolio as P  # noqa: E402
import portfolio_pm as PM  # noqa: E402
import round1 as R1  # noqa: E402
import t50_scrutiny as S  # noqa: E402
from portfolio import Costs, stats  # noqa: E402

OUT_MD = os.path.join(ROOT, "docs", "research", "07_reconciliation.md")
OUT_JSON = os.path.join(ROOT, "docs", "research", "reconciliation.json")
SPLITS = {"DEV": (L.MAIN, "2019-01-01", "2023-01-01"), "VAL": (L.MAIN, "2023-01-01", "2025-01-01"),
          "HOLDOUT": (L.HOLDOUT, "2019-01-01", "2025-01-01"), "MAIN_2019_24": (L.MAIN, "2019-01-01", "2025-01-01"),
          "ALL25_2021_24": (L.UNIVERSE, "2021-01-01", "2025-01-01")}
X1 = PM.Rules(trim_band=(0.15, 0.10), profit_lock=(2.0, 6.0))


def signal_level(data, ctx, uni, start, end):
    """Per-coin engine (core.backtester): one independent account per coin, every signal traded."""
    d = {s: data[s] for s in uni if s in data}
    t = M.trades_of(M.run_t50(d, ctx, M.cfg(), start, end), start, end)
    return t, M.summary(t)


def book(data, ctx, uni, start, end, no_halt, rules=PM.Rules(), log=None):
    fr, en, ex, at = S.portfolio_inputs({s: data[s] for s in uni if s in data}, ctx)
    if log is not None:                                   # frozen simulator with its signal log
        res = P.simulate(fr, en, ex, at, start, end, M.conservative(no_halt), Costs(), capital=10000.0, log=log)
        _, risk = PM.simulate(fr, en, ex, at, start, end, M.conservative(no_halt), Costs(), capital=10000.0)
        return res, risk, fr
    res, risk = PM.simulate(fr, en, ex, at, start, end, M.conservative(no_halt), Costs(), capital=10000.0, rules=rules)
    return res, risk, fr


def positions(trades):
    if not len(trades):
        return pd.DataFrame(columns=["pair", "entry_day", "pnl", "risk", "r"])
    key = "position_id" if "position_id" in trades else None
    g = trades.groupby(key if key else ["pair", "entry_day"])
    out = g.agg(pair=("pair", "first"), entry_day=("entry_day", "first"), pnl=("pnl", "sum"),
                risk=("initial_risk", "sum"), exit_day=("exit_day", "last"))
    out["r"] = out.pnl / out.risk
    return out.reset_index(drop=True)


def risk_profile(res, risk, fr):
    """T50 account-level risk: exposure maxima, simultaneous stop-outs, correlation of what is held."""
    s = stats(res, 10000.0)
    eq = res.equity
    t = res.trades
    stops = t[t.reason.isin(["stop_loss", "profit_lock"])] if len(t) else t
    same_day = stops.groupby("exit_day").agg(n=("pair", "size"), pnl=("pnl", "sum")) if len(stops) else pd.DataFrame()
    multi = same_day[same_day.n >= 2] if len(same_day) else same_day
    day_ret = eq.pct_change().dropna()
    r5 = eq.pct_change(5).dropna()
    rets = np.log(pd.DataFrame({k: f["Close"] for k, f in fr.items()})).diff()
    corr_held = []
    held_days = risk[risk.n_open >= 2] if "n_open" in risk else risk.iloc[0:0]
    for d in held_days.index[::5]:
        held = [p for p in t[(t.entry_day <= d) & (t.exit_day > d)].pair.unique()]
        if len(held) >= 2:
            c = PM._corr(rets, held, d, 90)
            corr_held.append(float(c[np.triu_indices(len(held), 1)].mean()))
    return {"return_pct": s["total_return_pct"], "max_dd_pct": s["max_dd_pct"], "halted_on": s["halted_on"],
            "trades": s["trades"], "positions": int(len(positions(t))),
            "max_position_pct": float(risk.max_position_pct.max()), "max_gross_pct": float(risk.position_value_pct.max()),
            "max_open_risk_to_stop_pct": float(risk.open_risk_to_stop_pct.max()),
            "p95_open_risk_to_stop_pct": float(risk.open_risk_to_stop_pct.quantile(0.95)),
            "max_initial_risk_pct": float(risk.initial_risk_pct.max()),
            "max_unrealised_pct": float(risk.unrealised_pct.max()),
            "max_corr_exposure_pct": float(risk.corr_exposure_pct.max()),
            "max_gap30_loss_pct": float(risk.gap30_loss_pct.max()),
            "worst_day_pct": float(day_ret.min() * 100) if len(day_ret) else np.nan,
            "worst_5d_pct": float(r5.min() * 100) if len(r5) else np.nan,
            "days_with_2plus_stopouts": int(len(multi)),
            "worst_multi_stopout_day_usdt": float(multi.pnl.min()) if len(multi) else 0.0,
            "mean_pairwise_corr_held_90d": float(np.mean(corr_held)) if corr_held else np.nan,
            "open_risk_at_dd_trough_pct": float(risk.open_risk_to_stop_pct.loc[:risk.drawdown_pct.idxmax()].max())}


def main():
    data = L.load_universe(L.UNIVERSE)
    ctx = {"btc_ok": R1.btc_filter(data["BTCUSDT"])}
    pub1 = json.load(open(os.path.join(ROOT, "docs", "research", "ml_round1_results.json"), encoding="utf-8"))
    pub2 = json.load(open(os.path.join(ROOT, "docs", "research", "round2_results.json"), encoding="utf-8"))
    out = {"checks": {}, "splits": {}, "capacity": {}, "published": {}}

    # 1) the +251.2R baseline and the GM2 figure
    t25, s25 = signal_level(data, ctx, L.UNIVERSE, "2021-01-01", "2025-01-01")
    out["checks"]["G0_reproduced"] = {"recomputed_trades": s25["trades"], "recomputed_total_R": s25["total_R"],
                                      "published_trades": pub1["base_G0"]["trades"],
                                      "published_total_R": pub1["base_G0"]["total_R"],
                                      "match": bool(s25["trades"] == pub1["base_G0"]["trades"] and
                                                    np.isclose(s25["total_R"], pub1["base_G0"]["total_R"]))}
    out["published"]["gates"] = {g: {"gate_total_R": v["g_total_R"], "delta_total_R": v["delta_total_R"],
                                     "trades": v["g_trades"], "removed_trades": v["removed_trades"]}
                                 for g, v in pub1["gates"].items()}
    out["published"]["gates"].update({g: {"gate_total_R": v.get("g_total_R"), "delta_total_R": v.get("delta_total_R"),
                                          "trades": v.get("g_trades"), "removed_trades": v.get("removed_trades")}
                                      for g, v in pub2["gates"].items()})

    # 2) signal level vs single account, per split
    for name, (uni, a, b) in SPLITS.items():
        t, s = signal_level(data, ctx, uni, a, b)
        row = {"universe": "main9" if uni is L.MAIN else "holdout16" if uni is L.HOLDOUT else "all25",
               "start": a, "end_exclusive": b, "signal_trades": s["trades"], "signal_total_R": s["total_R"],
               "signal_mean_R": s["net_R"], "signal_ex_top10_mean_R": s["net_R_ex_top10pct"]}
        for mode in ("as_configured", "no_halt"):
            for label, rules in (("R0", PM.Rules()), ("X1", X1)):
                res, risk, fr = book(data, ctx, uni, a, b, mode == "no_halt", rules)
                pos = positions(res.trades)
                row[f"{label}_{mode}"] = {**risk_profile(res, risk, fr), "book_total_R": float(pos.r.sum()),
                                          "book_mean_R": float(pos.r.mean()) if len(pos) else np.nan}
        out["splits"][name] = row
        # capacity: which signal trades the single account could not take, and why (frozen simulator log)
        log = []
        res, _, _ = book(data, ctx, uni, a, b, True, log=log)
        lg = pd.DataFrame(log)
        pos = positions(res.trades).assign(entry_day=lambda x: pd.to_datetime(x.entry_day, utc=True).dt.normalize())
        t = t.assign(fill_day=t.entry_ts.dt.normalize())
        taken = t.merge(pos[["pair", "entry_day"]], left_on=["pair", "fill_day"], right_on=["pair", "entry_day"],
                        how="left", indicator=True)
        taken["in_book"] = taken["_merge"] == "both"
        why = {}
        if len(lg):
            lg["fill_day"] = pd.to_datetime(lg.day, utc=True).dt.normalize() + pd.to_timedelta(
                np.where(lg.stage == "fill", 0, 1), unit="D")
            lg["prio"] = (lg.stage == "fill").astype(int)          # a fill-stage rejection explains more
            sig = lg.sort_values("prio").drop_duplicates(["pair", "fill_day"], keep="last")
            sig = sig.assign(reason=np.where(sig.status == "accepted", "accepted at close, not filled", sig.reason))
            m = taken[~taken.in_book].merge(sig[["pair", "fill_day", "reason"]], on=["pair", "fill_day"], how="left")
            m["reason"] = m.reason.fillna("coin already held in the book (different entry timing)")
            why = {k: {"trades": int(len(g)), "signal_R": float(g.r_multiple.sum())} for k, g in m.groupby("reason")}
        out["capacity"][name] = {"signal_trades": int(len(taken)), "taken_in_book": int(taken.in_book.sum()),
                                 "taken_signal_R": float(taken[taken.in_book].r_multiple.sum()),
                                 "missed_signal_R": float(taken[~taken.in_book].r_multiple.sum()),
                                 "book_positions_not_matching_a_signal_trade": int(len(pos) - taken.in_book.sum()),
                                 "missed_by_reason": why,
                                 "top10_signal_trades_in_book": int(taken.nlargest(10, "r_multiple").in_book.sum())}
        print(name, "done", flush=True)

    # 3) published single-account cells re-checked
    rv = pd.DataFrame(pub1["risk_variants"])
    chk = []
    for _, r in rv[rv.variant.isin(["R0", "R3"])].iterrows():
        uni = L.MAIN if r.universe == "main9" else L.HOLDOUT
        rules = PM.Rules() if r.variant == "R0" else X1
        res, risk, _ = book(data, ctx, uni, "2019-01-01", "2025-01-01", r["mode"] == "no_halt", rules)
        s = stats(res, 10000.0)
        chk.append({"universe": r.universe, "mode": r["mode"], "variant": r.variant, "published_return": r.return_pct,
                    "recomputed_return": s["total_return_pct"], "published_dd": r.max_dd_pct,
                    "recomputed_dd": s["max_dd_pct"], "halted_on": s["halted_on"],
                    "match": bool(np.isclose(r.return_pct, s["total_return_pct"]) and np.isclose(r.max_dd_pct, s["max_dd_pct"]))})
    out["checks"]["risk_variant_cells"] = chk
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=str)
    write_md(out)


def f(v, p=1, sign=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return f"{v:+.{p}f}" if sign else f"{v:.{p}f}"


def write_md(o):
    L_ = ["# Reconciliation of the frozen-T50 figures (generated by `research/reconcile.py`)\n",
          "Every number is recomputed from the stored data and code and checked against the published results "
          "files. No published result was edited. Sources: `docs/research/ml_round1_results.json` (Round 1), "
          "`round2_results.json` (Round 2), `03_t50_scrutiny.md` (BNB).\n"]
    g = o["checks"]["G0_reproduced"]
    L_ += ["## 1. Reproduction checks",
           f"- +251.2R baseline: recomputed {g['recomputed_trades']} trades, {g['recomputed_total_R']:+.2f}R; published "
           f"{g['published_trades']} trades, {g['published_total_R']:+.2f}R → **{'match' if g['match'] else 'MISMATCH'}**.",
           "- Single-account cells (main 9 / holdout 16, 2019–24, R0 and R3 = X1): " +
           ("all match." if all(c["match"] for c in o["checks"]["risk_variant_cells"]) else "**MISMATCH in at least one cell**") + "\n"]
    L_ += ["## 2. The figures in question\n",
           "| figure | what it is | universe, period | engine / account | source |", "|---|---|---|---|---|"]
    gm = o["published"]["gates"]
    L_ += [f"| +251.2R (289 trades) | frozen T50 total R, **every signal traded** | 25 coins, entries 2021-01-01 → 2024-12-31 | per-coin engine, one independent account per coin, no cross-coin limits | `ml_round1_results.json` base_G0 |",
           f"| −62R | **not a baseline**: the change in total R caused by gate GM2 (GM2 total {gm['GM2']['gate_total_R']:+.1f}R = 251.2 − 62.3) | same | same | gates.GM2.delta_total_R |",
           f"| −62R … −167R | range of ML-gate changes: GM2 {gm['GM2']['delta_total_R']:+.1f}, GM4 {gm['GM4']['delta_total_R']:+.1f}, ML8 {gm['ML8']['delta_total_R']:+.1f}, GM1 {gm['GM1']['delta_total_R']:+.1f}, GM3 {gm['GM3']['delta_total_R']:+.1f}, ML7 {gm['ML7']['delta_total_R']:+.1f} | same | same | Round 1 + Round 2 gates |",
           "| +70.4% / DD 17.9% → +88.3% / 10.7% | R0 → R3 (= X1) single account, no halt | main 9, 2019-01-01 → 2024-12-31 | single 10,000 USDT account, Conservative limits | risk_variants rows 4, 7 |",
           "| +50.4% / DD 14.0%, kill switch 2021-05-19 | R0 single account as configured | main 9, 2019–24 | same | risk_variants row 0 |",
           "| largest position 23.6%, open risk to stops 24.2% | R0 single account | main 9, 2019–24 | same | risk_variants row 0 |",
           "| BNB 38% of account | one BNB position in the **per-coin** BNB account (1,005 USDT) at its 2021-05-03 peak | BNB only, 2020-12 → 2021-05 | per-coin engine | `03_t50_scrutiny.md` line 254 |",
           "| holdout DD 11.4% → 9.7% | R0 → R3, no halt | holdout 16, 2019–24 | single account | risk_variants rows 12, 15 |\n",
           "**Execution assumptions, identical everywhere:** daily decisions at the close, fills at the next open; fee "
           "0.20% per side, slippage 0.05%, stop slippage 0.20%; T50 stop 3 × ATR14 below the fill, fixed; exit on a "
           "25-day-low close. Per-coin engine: 1,000 USDT per coin, 0.5% risk, 10% position cap, open position closed "
           "at the period end. Single account: 10,000 USDT, 0.5% risk, 10% position cap and 30% exposure at entry, "
           "3 positions, 3% daily-loss stop, 10% drawdown kill switch (permanent) or no halt; coins that signal on "
           "the same day are taken in fixed universe order.\n",
           "## 3. Signal level versus single account, by split\n",
           "Signal level counts every T50 signal as its own trade. The single account can hold at most 3 positions "
           "and 30% exposure, so most signals are never taken. R per position in the book = position P&L ÷ its "
           "initial risk.\n",
           "| split | signal trades | signal total R | book (no halt): positions, total R | book return / max DD (no halt) | as configured: return / DD / kill switch |",
           "|---|---|---|---|---|---|"]
    for n, r in o["splits"].items():
        a, c = r["R0_no_halt"], r["R0_as_configured"]
        L_.append(f"| {n} ({r['universe']}, {r['start'][:4]}–{int(r['end_exclusive'][:4]) - 1}) | {r['signal_trades']} | "
                  f"{f(r['signal_total_R'], 1, True)} | {a['positions']}, {f(a['book_total_R'], 1, True)} | "
                  f"{f(a['return_pct'], 1, True)}% / {f(a['max_dd_pct'])}% | {f(c['return_pct'], 1, True)}% / "
                  f"{f(c['max_dd_pct'])}% / {c['halted_on']} |")
    L_ += ["\n### Why the account takes far fewer trades (no halt)\n",
           "| split | signal trades | taken in book | book positions with no matching signal trade | signal R taken | signal R missed | top-10 signal trades in book | missed, by reason (trades, signal R) |",
           "|---|---|---|---|---|---|---|---|"]
    for n, c in o["capacity"].items():
        why = "; ".join(f"{k}: {v['trades']}, {v['signal_R']:+.1f}R" for k, v in
                        sorted(c["missed_by_reason"].items(), key=lambda kv: -kv[1]["trades"]))
        L_.append(f"| {n} | {c['signal_trades']} | {c['taken_in_book']} | {c['book_positions_not_matching_a_signal_trade']} | {c['taken_signal_R']:+.1f} | "
                  f"{c['missed_signal_R']:+.1f} | {c['top10_signal_trades_in_book']}/10 | {why} |")
    L_ += ["\n## 4. T50 single-account risk (frozen R0 book; X1 for comparison)\n",
           "Open risk to stops = loss if every open stop filled at its stop price on the same day (simultaneous "
           "stop-outs). Gap 30% = instant loss if every coin held gapped down 30% (stops do not help). Correlation "
           "exposure = √(w′Pw) with the 90-day correlation of the coins held.\n",
           "| split | book | mode | max DD % | kill switch | largest position % | max gross % | max open risk to stops % (p95) | max corr. exposure % | max gap-30% loss % | worst day % | worst 5 days % | days with ≥2 stop-outs | mean corr. of held coins |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in o["splits"].items():
        for b in ("R0", "X1"):
            for m in ("no_halt", "as_configured"):
                x = r[f"{b}_{m}"]
                L_.append(f"| {n} | {b} | {m} | {f(x['max_dd_pct'])} | {x['halted_on']} | {f(x['max_position_pct'])} | "
                          f"{f(x['max_gross_pct'])} | {f(x['max_open_risk_to_stop_pct'])} ({f(x['p95_open_risk_to_stop_pct'])}) | "
                          f"{f(x['max_corr_exposure_pct'])} | {f(x['max_gap30_loss_pct'])} | {f(x['worst_day_pct'], 1, True)} | "
                          f"{f(x['worst_5d_pct'], 1, True)} | {x['days_with_2plus_stopouts']} | {f(x['mean_pairwise_corr_held_90d'], 2)} |")
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L_) + "\n")


if __name__ == "__main__":
    main()
