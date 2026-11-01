"""Round 1 of docs/research/RESEARCH_PROGRAMME.md (pre-registered in commit b9c23ba).

    python research/round1.py            -> docs/research/02_round1_results.md

Candidates' signals are fed into the unchanged OF2 engine (core/backtester.run_backtest), so costs, stops,
sizing, the volume cap and the cost-to-risk filter are identical to the paper trader. Data on or after
2025-01-01 is dropped right after loading: the consumed TEST window is never seen.
"""
import copy
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from core import validation as V  # noqa: E402
from core.backtester import run_backtest  # noqa: E402
from core.core_signals import generate_signals  # noqa: E402
from core.history import DEFAULT_UNIVERSE, load_dataset  # noqa: E402
from utils.config import build_settings, load_config  # noqa: E402

CUTOFF = pd.Timestamp("2025-01-01", tz="UTC")
DEV, VAL, ALL = ("2019-01-01", "2023-01-01"), ("2023-01-01", "2025-01-01"), ("2019-01-01", "2025-01-01")
HOLDOUT = ["ETCUSDT", "LINKUSDT", "NEOUSDT", "QTUMUSDT", "IOTAUSDT", "ONTUSDT", "ICXUSDT", "WAVESUSDT", "BATUSDT",
           "VETUSDT", "ZILUSDT", "NULSUSDT", "ZECUSDT", "DASHUSDT", "XMRUSDT", "OMGUSDT"]
FAMILY = ("T50", "T20", "T50B")
OUT = os.path.join(ROOT, "docs", "research", "02_round1_results.md")
STOPS = {"stop_loss", "trailing_stop"}
N_RANDOM = 200


# ------------------------------------------------------------------ data and config
def load(symbols, interval="1d"):
    out = {}
    for s in symbols:
        df = load_dataset(s, interval)
        df = df[pd.to_datetime(df["timestamp"], utc=True) < CUTOFF].reset_index(drop=True)
        assert pd.to_datetime(df["timestamp"], utc=True).max() < CUTOFF
        out[s] = df
    return out


def config(fee=None, standard_exits=False):
    bt, _ = build_settings(load_config(), "Conservative", {"interval": "1d"})
    bt.initial_capital = 1000.0
    if fee is not None:
        bt.fee_rate = fee
    if not standard_exits:
        bt.exit_rules.stop_atr_mult = 3.0
        bt.exit_rules.tp_atr_mult = 1000.0
        bt.exit_rules.use_trailing = False
        bt.exit_rules.max_hold_bars = 0
    return bt


# ------------------------------------------------------------------ market context (causal)
def btc_filter(btc):
    c = btc.set_index(pd.to_datetime(btc["timestamp"], utc=True))["Close"]
    return c > c.rolling(100).mean()                       # known at that day's close


def rs_basket(data, btc_ok, top=3, lookback=28):
    """Per date (Mondays only): set of coins in the RS3 basket. Uses closes up to that date."""
    closes = pd.DataFrame({s: d.set_index(pd.to_datetime(d["timestamp"], utc=True))["Close"] for s, d in data.items()})
    mom = closes / closes.shift(lookback) - 1
    rank = mom.rank(axis=1, ascending=False, method="first")
    member = (rank <= top) & (mom > 0)
    member = member & btc_ok.reindex(member.index).fillna(False).astype(bool).to_numpy()[:, None]
    monday = member.index.dayofweek == 0
    return member, monday


# ------------------------------------------------------------------ signals
_FRAMES = {}


def base_frame(seg):
    """Indicators, ATR(14) and regime label from the existing code (cached: identical input -> identical frame)."""
    key = (float(seg["Close"].iloc[0]), str(seg["timestamp"].iloc[0]), str(seg["timestamp"].iloc[-1]), len(seg))
    if key not in _FRAMES:
        sig = generate_signals(seg, strategy="trend_following")
        sig["confidence"] = 1.0
        _FRAMES[key] = sig
    return _FRAMES[key].copy()


def signals(name, seg, ctx, rng=None, p=None):
    sig = base_frame(seg)
    ts = pd.to_datetime(seg["timestamp"], utc=True)
    c = seg["Close"]
    if name == "REF":
        return sig                                                   # existing trend_following decisions
    if name in FAMILY or name.startswith("RAND_T"):
        n = 20 if name in ("T20", "RAND_T20") else 50
        entry = c > c.rolling(n).max().shift(1)
        exit_ = c < c.rolling(n // 2).min().shift(1)
        if name == "T50B":
            entry &= ctx["btc_ok"].reindex(ts).fillna(False).astype(bool).to_numpy()
        if name.startswith("RAND_T"):                                # random entries, same exits and stops
            entry = pd.Series(rng.random(len(seg)) < p, index=seg.index)
    elif name in ("RS3", "RAND_RS"):
        member = ctx["member"][ctx["pair"]].reindex(ts).fillna(False).astype(bool).to_numpy()
        monday = (ts.dt.dayofweek == 0).to_numpy()
        if name == "RAND_RS":                                        # 3 random coins on BTC-filter-on Mondays
            member = ctx["rand_member"][ctx["pair"]].reindex(ts).fillna(False).astype(bool).to_numpy()
        entry = pd.Series(member & monday, index=seg.index)
        exit_ = pd.Series(monday & ~member, index=seg.index)
    else:
        raise ValueError(name)
    sig["entry"] = entry.fillna(False).to_numpy(bool)
    sig["exit"] = exit_.fillna(False).to_numpy(bool)
    sig["strategy"] = name
    return sig


def run(name, data, ctx, cfg, start, end, **kw):
    out = {}
    for pair, df in data.items():
        seg = V.segment(df, start, end)
        ts = pd.to_datetime(seg["timestamp"], utc=True)
        if seg.empty or (ts >= pd.Timestamp(start, tz="UTC")).sum() < 50:
            continue
        c = copy.deepcopy(cfg)
        c.strategy, c.trade_from = "trend_following", start
        sig = signals(name, seg, {**ctx, "pair": pair}, **kw)
        out[pair] = run_backtest(seg, c, pair, signals=sig)
    return out


# ------------------------------------------------------------------ metrics
def gross_r(results, cfg):
    t = pd.concat([r.trades for r in results.values() if len(r.trades)], ignore_index=True) \
        if any(len(r.trades) for r in results.values()) else pd.DataFrame()
    if t.empty:
        return float("nan")
    raw_in = t["entry_price"] / (1 + cfg.slippage_pct)
    raw_out = t["exit_price"] / (1 - np.where(t["exit_reason"].isin(STOPS), cfg.stop_slippage_pct, cfg.slippage_pct))
    return float((t["qty"] * (raw_out - raw_in) / t["risk_amount"]).mean())


def portfolio(results, start):
    eq, bh = [], []
    for pair, r in results.items():
        e = r.equity.assign(t=pd.to_datetime(r.equity["time"], utc=True)).set_index("t")["equity"]
        e = e[e.index >= pd.Timestamp(start, tz="UTC")]
        eq.append(e.rename(pair))
    if not eq:
        return {}
    E = pd.concat(eq, axis=1).ffill().bfill().sum(axis=1)
    ret = E.pct_change().dropna()
    dd = (E / E.cummax() - 1).min()
    return {"portfolio_return_pct": float((E.iloc[-1] / E.iloc[0] - 1) * 100), "portfolio_max_dd_pct": float(-dd * 100),
            "sharpe": float(ret.mean() / ret.std() * np.sqrt(365)) if ret.std() > 0 else float("nan"),
            "avg_exposure_pct": float(np.mean([r.equity["in_position"].mean() for r in results.values()]) * 100)}


def summarise(results, cfg, start):
    p = V.pooled(results)
    return {"trades": p["trades"], "net_R": p["expectancy_r"], "ci_low": p["ci_low"], "gross_R": gross_r(results, cfg),
            "pf": p["profit_factor"], "win_pct": p["win_rate_pct"], "breadth": p["breadth"],
            "coins": p["eligible_coins"], "wo_best": p["exp_r_without_best_coin"],
            "mean_net_ret_pct": p["mean_net_return_pct"], "mean_bh_pct": p["mean_buy_hold_pct"],
            "worst_coin_dd_pct": p["worst_max_dd_pct"], **portfolio(results, start),
            "_by_regime": p["by_regime"], "_by_year": p["by_year"], "_per_coin": p["per_coin"]}


def p3(s):
    f = []
    if s["trades"] < 100: f.append(f"{s['trades']} trades < 100")
    if not s["net_R"] > 0: f.append(f"expectancy {s['net_R']:+.3f}R <= 0")
    if not s["ci_low"] > 0: f.append(f"CI low {s['ci_low']:+.3f}R <= 0")
    if not s["pf"] >= 1.10: f.append(f"PF {s['pf']:.2f} < 1.10")
    if not s["breadth"] >= 0.60: f.append(f"breadth {s['breadth']:.0%} < 60%")
    if not s["wo_best"] > 0: f.append(f"w/o best coin {s['wo_best']:+.3f}R <= 0")
    return f


def holdout_check(s):
    f = []
    if not s["net_R"] > 0: f.append(f"holdout expectancy {s['net_R']:+.3f}R <= 0")
    if not s["pf"] >= 1.10: f.append(f"holdout PF {s['pf']:.2f} < 1.10")
    if not s["breadth"] >= 0.50: f.append(f"holdout breadth {s['breadth']:.0%} < 50%")
    return f


# ------------------------------------------------------------------ information tests
def info_reversal(data):
    rows = []
    for pair, df in data.items():
        d = df[pd.to_datetime(df["timestamp"], utc=True) < pd.Timestamp(DEV[1], tz="UTC")].reset_index(drop=True)
        lr = np.log(d["Close"]).diff()
        thr = lr.rolling(365, min_periods=180).quantile(0.05).shift(1)
        ev = lr < thr
        o = d["Open"]
        for h in (1, 3, 5):
            fwd = (o.shift(-1 - h) / o.shift(-1) - 1)                # enter next open, exit h days later at open
            rows.append({"pair": pair, "h": h, "events": int(ev.sum()), "mean_fwd_pct": float(fwd[ev].mean() * 100),
                         "uncond_pct": float(fwd.mean() * 100)})
    t = pd.DataFrame(rows)
    return t.groupby("h").agg(events=("events", "sum"), mean_fwd_pct=("mean_fwd_pct", "mean"),
                              uncond_pct=("uncond_pct", "mean")).reset_index()


def info_leadlag():
    btc = load(["BTCUSDT"], "1h")["BTCUSDT"]
    b = btc.set_index(pd.to_datetime(btc["timestamp"], utc=True))["Close"].pct_change()
    rows = []
    for pair in [s for s in DEFAULT_UNIVERSE if s != "BTCUSDT"]:
        a = load([pair], "1h")[pair]
        a = a.set_index(pd.to_datetime(a["timestamp"], utc=True))
        o = a["Open"]
        nxt = (o.shift(-2) / o.shift(-1) - 1)                         # next full hour, from next open
        j = pd.DataFrame({"b": b, "n": nxt}).dropna()
        j = j[j.index < pd.Timestamp(DEV[1], tz="UTC")]
        rows.append({"pair": pair, "after_btc_up1%": j.loc[j.b > 0.01, "n"].mean() * 100, "n_up": int((j.b > 0.01).sum()),
                     "after_btc_down1%": j.loc[j.b < -0.01, "n"].mean() * 100, "n_down": int((j.b < -0.01).sum()),
                     "unconditional": j["n"].mean() * 100})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ report helpers
def md(df):
    d = df.copy()
    for col in d.columns:
        if d[col].dtype.kind == "f":
            d[col] = d[col].map(lambda v: "nan" if pd.isna(v) else f"{v:+.3f}" if abs(v) < 10 else f"{v:+.1f}")
    return ("| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns) + "\n" +
            "\n".join("| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False)))


def row(name, period, s):
    return {"candidate": name, "period": period, **{k: v for k, v in s.items() if not k.startswith("_")}}


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    data, hold = load(DEFAULT_UNIVERSE), load(HOLDOUT)
    btc_ok = btc_filter(data["BTCUSDT"])
    member, _ = rs_basket(data, btc_ok)
    hmember, _ = rs_basket(hold, btc_ok)
    ctx, hctx = {"btc_ok": btc_ok, "member": member}, {"btc_ok": btc_ok, "member": hmember}
    cfg = config()
    rows, notes, extra = [], [], {}

    # 1) DEV selection for the trend family
    dev = {n: summarise(run(n, data, ctx, cfg, *DEV), cfg, DEV[0]) for n in FAMILY}
    for n, s in dev.items():
        rows.append(row(n, "DEV", s))
    ok = {n: s for n, s in dev.items() if s["trades"] >= 100}
    pick = max(ok, key=lambda n: ok[n]["net_R"]) if ok else None
    notes.append(f"Trend family DEV pick: **{pick}** (highest DEV net expectancy with >= 100 trades).")
    rs_dev = summarise(run("RS3", data, ctx, cfg, *DEV), cfg, DEV[0])
    rows.append(row("RS3", "DEV", rs_dev))
    ref_cfg = config(standard_exits=True)
    rows.append(row("REF", "DEV", summarise(run("REF", data, ctx, ref_cfg, *DEV), ref_cfg, DEV[0])))
    print("DEV done", round(time.time() - t0), flush=True)

    # 2) walk-forward
    wf = []
    for y in (2021, 2022, 2023, 2024):
        tr = (f"{y - 2}-01-01", f"{y}-01-01")
        sc = {n: summarise(run(n, data, ctx, cfg, *tr), cfg, tr[0]) for n in FAMILY}
        cand = {n: s for n, s in sc.items() if s["trades"] >= 50}
        sel = max(cand, key=lambda n: cand[n]["net_R"]) if cand else None
        te = (f"{y}-01-01", f"{y + 1}-01-01")
        s = summarise(run(sel, data, ctx, cfg, *te), cfg, te[0]) if sel else None
        r = summarise(run("RS3", data, ctx, cfg, *te), cfg, te[0])
        wf.append({"year": y, "trend_selected": sel or "-", "trend_trades": s["trades"] if s else 0,
                   "trend_net_R": s["net_R"] if s else float("nan"), "RS3_trades": r["trades"], "RS3_net_R": r["net_R"]})
    wf = pd.DataFrame(wf)
    wf_trend = int((wf["trend_net_R"] > 0).sum())
    wf_rs = int((wf["RS3_net_R"] > 0).sum())
    print("WF done", round(time.time() - t0), flush=True)

    # 3) VAL once, holdout once, cost stress
    verdicts = {}
    for name in [x for x in (pick, "RS3") if x]:
        v = summarise(run(name, data, ctx, cfg, *VAL), cfg, VAL[0])
        h = summarise(run(name, hold, hctx, cfg, *ALL), cfg, ALL[0])
        st = summarise(run(name, data, ctx, config(fee=0.003), *VAL), config(fee=0.003), VAL[0])
        rows += [row(name, "VAL", v), row(name, "HOLDOUT 2019-24", h), row(name, "VAL @0.30% fee", st)]
        extra[name] = {"val": v, "hold": h}
        verdicts[name] = p3(v) + holdout_check(h)
    rows.append(row("REF", "VAL (already seen)", summarise(run("REF", data, ctx, ref_cfg, *VAL), ref_cfg, VAL[0])))
    print("VAL/holdout done", round(time.time() - t0), flush=True)

    # 4) random-entry benchmark on DEV+VAL (main universe)
    bench = {}
    for name, rname in [(pick, "RAND_T"), ("RS3", "RAND_RS")]:
        if not name:
            continue
        real = summarise(run(name, data, ctx, cfg, *ALL), cfg, ALL[0])
        draws = []
        rng = np.random.default_rng(11)
        if rname == "RAND_T":
            n = 20 if name == "T20" else 50
            probs = {}
            for pair, df in data.items():
                c = df["Close"]
                e = c > c.rolling(n).max().shift(1)
                if name == "T50B":
                    e &= btc_ok.reindex(pd.to_datetime(df["timestamp"], utc=True)).fillna(False).to_numpy()
                probs[pair] = float(e.mean())
            for _ in range(N_RANDOM):
                res = {}
                for pair, df in data.items():
                    seg = V.segment(df, *ALL)
                    c = copy.deepcopy(cfg)
                    c.strategy, c.trade_from = "trend_following", ALL[0]
                    res[pair] = run_backtest(seg, c, pair, signals=signals("RAND_T50" if n == 50 else "RAND_T20", seg,
                                                                            ctx, rng=rng, p=probs[pair]))
                draws.append(V.pooled(res)["expectancy_r"])
        else:
            on = btc_ok.reindex(member.index).fillna(False)
            for _ in range(N_RANDOM):
                rm = pd.DataFrame(False, index=member.index, columns=member.columns)
                for d in member.index[(member.index.dayofweek == 0) & on.to_numpy()]:
                    rm.loc[d, rng.choice(member.columns, 3, replace=False)] = True
                draws.append(V.pooled(run("RAND_RS", data, {**ctx, "rand_member": rm}, cfg, *ALL))["expectancy_r"])
        draws = np.array([d for d in draws if np.isfinite(d)])
        p95 = float(np.percentile(draws, 95))
        bench[name] = {"real_net_R": real["net_R"], "real_trades": real["trades"], "random_mean": float(draws.mean()),
                       "random_p95": p95, "percentile": float((draws < real["net_R"]).mean() * 100)}
        if not real["net_R"] > p95:
            verdicts[name].append(f"net {real['net_R']:+.3f}R not above random-entry 95th pct {p95:+.3f}R")
        print("bench", name, bench[name], round(time.time() - t0), flush=True)
    if pick and wf_trend < 3:
        verdicts[pick].append(f"walk-forward {wf_trend}/4 positive < 3")
    if wf_rs < 3:
        verdicts["RS3"].append(f"walk-forward {wf_rs}/4 positive < 3")

    # 5) information tests
    rev = info_reversal(data)
    ll = info_leadlag()

    # report
    cols = ["candidate", "period", "trades", "gross_R", "net_R", "ci_low", "pf", "win_pct", "breadth", "coins", "wo_best",
            "portfolio_return_pct", "portfolio_max_dd_pct", "sharpe", "avg_exposure_pct", "mean_bh_pct"]
    summary = pd.DataFrame(rows)[cols]
    txt = [f"# Round 1 results ({pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC, {time.time() - t0:.0f}s)\n",
           "Pre-registered in `docs/research/RESEARCH_PROGRAMME.md` (commit b9c23ba). Unchanged engine and costs; "
           "Conservative risk; daily bars; data >= 2025-01-01 never loaded.\n",
           "Columns: R values are per trade; `breadth` = share of coins (>= 10 trades) with positive expectancy; "
           "portfolio = sum of the per-coin 1,000 USDT accounts; `mean_bh_pct` = mean buy-and-hold return per coin.\n",
           "## Summary\n", md(summary), "\n", "\n".join(f"- {n}" for n in notes), "\n",
           "## Walk-forward (trend variant re-selected on the 2 prior years; RS3 fixed)\n", md(wf),
           f"\nTrend: {wf_trend}/4 positive. RS3: {wf_rs}/4 positive.\n",
           "## Random-entry benchmark (DEV+VAL, main universe, 200 runs, same exits and stops)\n",
           md(pd.DataFrame(bench).T.reset_index().rename(columns={"index": "candidate"})), "\n",
           "## Verdicts\n"]
    for n, f in verdicts.items():
        txt.append(f"- **{n}: {'RESEARCH-ELIGIBLE' if not f else 'FAILED'}**" + ("" if not f else ": " + "; ".join(f)))
    for n, e in extra.items():
        txt += [f"\n## {n} detail\n", "VAL by regime at entry:\n", md(e["val"]["_by_regime"].reset_index()),
                "\nVAL by year:\n", md(e["val"]["_by_year"].reset_index()),
                "\nVAL per coin:\n", md(e["val"]["_per_coin"]),
                "\nHoldout per coin:\n", md(e["hold"]["_per_coin"])]
    txt += ["\n## Information test H3: forward return after an extreme daily drop (DEV only)\n",
            "Event: daily log return below the coin's trailing-365d 5th percentile. Entry next open; % return.\n",
            md(rev), "\nRound-trip cost is ~0.5%.\n",
            "\n## Information test H4: altcoin next-hour return after BTC hourly moves (DEV only, %)\n", md(ll),
            "\nRound-trip cost is ~0.5%.\n"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(txt))
    with open(os.path.join(ROOT, "state", "research", "round1_verdicts.json"), "w", encoding="utf-8") as f:
        json.dump({"pick": pick, "verdicts": verdicts, "bench": bench}, f, indent=1, default=str)
    print("\n".join(txt)[:8000])


if __name__ == "__main__":
    main()
