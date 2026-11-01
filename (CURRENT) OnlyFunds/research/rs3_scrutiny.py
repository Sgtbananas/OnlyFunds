"""RS3 scrutiny: is RS3's positive record an edge, or one or two exceptional trades? (descriptive; rules frozen)

    python research/rs3_scrutiny.py      -> docs/research/05_rs3_scrutiny.md, docs/research/rs3_scrutiny.json

RS3 is used exactly as pre-registered in RESEARCH_PROGRAMME.md and frozen for shadow tracking. No rule is changed,
nothing is tuned, data on or after 2025-01-01 is never loaded, and the shadow tracker is not touched.
Random benchmark: on every BTC-filter-on Monday, 3 random coins of the same universe are held until the next
Monday (the same exits, stops, sizing and costs). N draws from env RS3_N_RANDOM (default 500), seed 3303.
"""
import copy
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import ml_lab as L  # noqa: E402
import round1 as R1  # noqa: E402
import portfolio_variants as PV  # noqa: E402
from portfolio import Costs, stats  # noqa: E402
from core import validation as V  # noqa: E402
from core.backtester import run_backtest  # noqa: E402

OUT_MD = os.path.join(ROOT, "docs", "research", "05_rs3_scrutiny.md")
OUT_JSON = os.path.join(ROOT, "docs", "research", "rs3_scrutiny.json")
N_RANDOM = int(os.environ.get("RS3_N_RANDOM", "500"))
SAMPLES = {"main9 DEV 2019-22": ("main", R1.DEV), "main9 VAL 2023-24": ("main", R1.VAL),
           "main9 2019-24": ("main", R1.ALL), "holdout16 2019-24": ("holdout", R1.ALL)}


def cfg():
    c = R1.config()
    c.close_at_end = True
    return c


def run_rs(data, member, btc_ok, start, end, name="RS3", rand_member=None):
    out = {}
    for pair, df in data.items():
        seg = V.segment(df, start, end)
        ts = pd.to_datetime(seg["timestamp"], utc=True)
        if seg.empty or (ts >= pd.Timestamp(start, tz="UTC")).sum() < 50:
            continue
        c = cfg()
        c.strategy, c.trade_from = "trend_following", start
        sig = R1.signals(name, seg, {"btc_ok": btc_ok, "member": member, "rand_member": rand_member, "pair": pair})
        out[pair] = run_backtest(seg, c, pair, signals=sig)
    parts = [r.trades.assign(pair=p) for p, r in out.items() if len(r.trades)]
    if not parts:
        return pd.DataFrame(columns=["pair", "r_multiple", "pnl", "entry_ts", "week", "year"])
    t = pd.concat(parts, ignore_index=True)
    t["entry_ts"] = pd.to_datetime(t["entry_time"], utc=True)
    t = t[(t.entry_ts >= pd.Timestamp(start, tz="UTC")) & (t.entry_ts < pd.Timestamp(end, tz="UTC"))]
    t["week"] = t.entry_ts.dt.strftime("%G-%V")
    t["year"] = t.entry_ts.dt.year
    return t.reset_index(drop=True)


def ci(t):
    if len(t) < 2:
        return (np.nan, np.nan)
    return L.week_cluster_ci(t.r_multiple.to_numpy(), t.week.to_numpy())


def concentration(t):
    r = t.r_multiple.sort_values(ascending=False)
    total = r.sum()
    row = {"trades": len(t), "total_R": total, "mean_R": r.mean(), "median_R": r.median(),
           "ci_lo": ci(t)[0], "ci_hi": ci(t)[1], "win_pct": (t.pnl > 0).mean() * 100}
    for k in (1, 2, 5):
        rest = t.loc[r.index[k:]]
        row[f"top{k}_share_pct"] = r.iloc[:k].sum() / total * 100 if total > 0 else np.nan
        row[f"mean_ex_top{k}"] = rest.r_multiple.mean()
        row[f"ci_lo_ex_top{k}"] = ci(rest)[0]
    k10 = int(np.ceil(len(t) * 0.10))
    rest = t.loc[r.index[k10:]]
    row["mean_ex_top10pct"], row["ci_lo_ex_top10pct"] = rest.r_multiple.mean(), ci(rest)[0]
    return row


def random_baskets(data, member, btc_ok, start, end, n, seed):
    rng = np.random.default_rng(seed)
    on = btc_ok.reindex(member.index).fillna(False).to_numpy(bool)
    mondays = member.index[(member.index.dayofweek == 0) & on]
    out = []
    for _ in range(n):
        rm = pd.DataFrame(False, index=member.index, columns=member.columns)
        for d in mondays:
            avail = [c for c in member.columns if np.isfinite(data_close(data, c, d))]
            if len(avail) >= 3:
                rm.loc[d, rng.choice(avail, 3, replace=False)] = True
        t = run_rs(data, member, btc_ok, start, end, "RAND_RS", rm)
        out.append({"mean_R": t.r_multiple.mean() if len(t) else np.nan, "total_R": t.r_multiple.sum(),
                    "trades": len(t)})
    return pd.DataFrame(out)


_CLOSE = {}


def data_close(data, c, d):
    if c not in _CLOSE:
        df = data[c]
        _CLOSE[c] = pd.Series(df["Close"].to_numpy(float), index=pd.to_datetime(df["timestamp"], utc=True))
    return float(_CLOSE[c].get(d, np.nan))


def md(df, idx=False):
    d = df.reset_index() if idx else df.copy()
    for col in d.columns:
        if d[col].dtype.kind == "f":
            d[col] = d[col].map(lambda v: "nan" if pd.isna(v) else f"{v:+.3f}" if abs(v) < 10 else f"{v:+.1f}")
    return ("| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns) + "\n" +
            "\n".join("| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False)))


def main():
    data = L.load_universe()
    btc_ok = R1.btc_filter(data["BTCUSDT"])
    import t50_scrutiny as S
    btc_lab = S.btc_state(data["BTCUSDT"])
    unis = {"main": {s: data[s] for s in L.MAIN}, "holdout": {s: data[s] for s in L.HOLDOUT}}
    members = {k: R1.rs_basket(u, btc_ok)[0] for k, u in unis.items()}
    conc, loco, years, states, bench, trades = [], [], [], [], {}, {}
    for label, (u, (s, e)) in SAMPLES.items():
        t = run_rs(unis[u], members[u], btc_ok, s, e)
        trades[label] = t
        conc.append({"sample": label, **concentration(t)})
    for label in ("main9 2019-24", "holdout16 2019-24"):
        t, u = trades[label], SAMPLES[label][0]
        for coin in sorted(t.pair.unique()):
            sub = t[t.pair != coin]
            loco.append({"sample": label, "without": coin, "trades": len(sub), "mean_R": sub.r_multiple.mean(),
                         "ci_lo": ci(sub)[0]})
        for y, sub in t.groupby("year"):
            years.append({"sample": label, "year": y, "trades": len(sub), "total_R": sub.r_multiple.sum(),
                          "mean_R": sub.r_multiple.mean(), "ci_lo": ci(sub)[0]})
        st = t.assign(state=btc_lab.reindex(t.entry_ts.dt.normalize() - pd.Timedelta(days=1)).fillna("n/a").to_numpy())
        for k, sub in st.groupby("state"):
            states.append({"sample": label, "btc_state": k, "trades": len(sub), "total_R": sub.r_multiple.sum(),
                           "mean_R": sub.r_multiple.mean(), "ci_lo": ci(sub)[0]})
        rb = random_baskets(unis[u], members[u], btc_ok, *R1.ALL, N_RANDOM, 3303)
        bench[label] = {"RS3_mean_R": float(t.r_multiple.mean()), "RS3_total_R": float(t.r_multiple.sum()),
                        "random_mean_R_median": float(rb.mean_R.median()), "random_mean_R_p95": float(rb.mean_R.quantile(.95)),
                        "percentile_mean_R": float((rb.mean_R < t.r_multiple.mean()).mean() * 100),
                        "p_mean_R": float(((rb.mean_R >= t.r_multiple.mean()).sum() + 1) / (len(rb) + 1)),
                        "random_total_R_median": float(rb.total_R.median()),
                        "percentile_total_R": float((rb.total_R < t.r_multiple.sum()).mean() * 100),
                        "random_trades_mean": float(rb.trades.mean()), "RS3_trades": int(len(t)), "draws": int(len(rb))}
        print(label, bench[label], flush=True)
    # single account (Conservative limits) for RS3, main 9
    fr, en, ex, at = {}, {}, {}, {}
    for pair, df in unis["main"].items():
        sig = R1.signals("RS3", df, {"btc_ok": btc_ok, "member": members["main"], "pair": pair})
        idx = pd.to_datetime(df["timestamp"], utc=True)
        fr[pair] = df.set_index(idx)[["Open", "High", "Low", "Close", "Volume"]]
        en[pair] = pd.Series(sig["entry"].to_numpy(bool), index=idx)
        ex[pair] = pd.Series(sig["exit"].to_numpy(bool), index=idx)
        at[pair] = pd.Series(sig["ATR"].to_numpy(float), index=idx)
    import ml_round1 as M
    port = {}
    for mode in ("as_configured", "no_halt"):
        res, _ = PV.simulate(fr, en, ex, at, *R1.ALL, M.conservative(mode == "no_halt"), Costs(), capital=10000.0)
        port[mode] = stats(res, 10000.0)
    top = trades["main9 2019-24"].nlargest(5, "r_multiple")[["pair", "entry_time", "exit_time", "r_multiple"]]
    out = {"concentration": conc, "leave_one_coin_out": loco, "years": years, "btc_state": states,
           "random_benchmark": bench, "portfolio_main9": port, "n_random": N_RANDOM}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    L_ = ["# RS3 scrutiny (rules frozen; data 2019–2024 only; descriptive)\n",
          f"Reproduce: `python research/rs3_scrutiny.py` ({N_RANDOM} random-basket draws).\n",
          "## Concentration in the largest trades\n", md(pd.DataFrame(conc)),
          "\n**Five largest trades, main 9, 2019–24:**\n", md(top),
          "\n## Random 3-coin baskets on the same Mondays (same exits, stops, sizing, costs)\n",
          md(pd.DataFrame(bench).T, idx=True),
          "\n## Leave one coin out\n", md(pd.DataFrame(loco)),
          "\n## Years\n", md(pd.DataFrame(years)),
          "\n## BTC market state at the decision\n", md(pd.DataFrame(states)),
          "\n## Single account, Conservative limits, main 9, 2019–24\n", md(pd.DataFrame(port).T, idx=True)]
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L_) + "\n")
    return out


if __name__ == "__main__":
    main()
