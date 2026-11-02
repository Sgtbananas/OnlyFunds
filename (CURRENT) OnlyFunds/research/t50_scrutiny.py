"""Scrutiny of T50 with its rules frozen exactly as pre-registered (docs/research/RESEARCH_PROGRAMME.md).

    python research/t50_scrutiny.py      -> docs/research/03_t50_scrutiny.md (+ figures in docs/research/fig/)

No rule is changed. Data on or after 2025-01-01 is never loaded (consumed TEST window).
Sections: outliers, leave-one-coin-out, years/regimes, drawdown reconciliation, matched random-entry
benchmark (1,000 runs), holdout with listing/delisting, single-account economics and cost sensitivity.
"""
import copy
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
import round1 as R1  # noqa: E402
from portfolio import Costs, simulate, stats  # noqa: E402
from core import validation as V  # noqa: E402
from core.backtester import run_backtest  # noqa: E402
from core.regime import detect  # noqa: E402
from core.risk_manager import RiskLimits  # noqa: E402

OUT = os.path.join(ROOT, "docs", "research", "03_t50_scrutiny.md")
FIG = os.path.join(ROOT, "docs", "research", "fig")
ALL, DEV, VAL = R1.ALL, R1.DEV, R1.VAL
N_RANDOM = int(os.environ.get("T50_N_RANDOM", "1000"))
SEED = 2026


def md(df, idx=False):
    d = df.reset_index() if idx else df.copy()
    for col in d.columns:
        if d[col].dtype.kind == "f":
            d[col] = d[col].map(lambda v: "nan" if pd.isna(v) else f"{v:+.3f}" if abs(v) < 10 else f"{v:+.1f}")
    return ("| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns) + "\n" +
            "\n".join("| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False)))


def cfg(close_at_end=True, fee=None, slip=None, stop_slip=None):
    c = R1.config(fee=fee)
    c.close_at_end = close_at_end
    if slip is not None:
        c.slippage_pct = slip
    if stop_slip is not None:
        c.stop_slippage_pct = stop_slip
    return c


def trades_of(results):
    t = pd.concat([r.trades.assign(pair=p) for p, r in results.items() if len(r.trades)], ignore_index=True)
    t["entry_ts"] = pd.to_datetime(t["entry_time"], utc=True)
    t["year"] = t["entry_ts"].dt.year
    return t


def basic(t):
    if len(t) == 0:
        return {"trades": 0, "net_R": np.nan, "pf": np.nan, "total_R": 0.0, "total_pnl": 0.0, "win_pct": np.nan,
                "median_R": np.nan}
    gl = -t.loc[t.pnl <= 0, "pnl"].sum()
    return {"trades": len(t), "net_R": t.r_multiple.mean(), "median_R": t.r_multiple.median(),
            "pf": t.loc[t.pnl > 0, "pnl"].sum() / gl if gl > 0 else np.inf, "win_pct": (t.pnl > 0).mean() * 100,
            "total_R": t.r_multiple.sum(), "total_pnl": t.pnl.sum()}


def ci_low(r):
    return V.bootstrap_lower(np.asarray(r, float))[0]


# ------------------------------------------------------------------ 1) outliers
def outliers(t, label):
    rows = []
    s = t.sort_values("r_multiple", ascending=False)
    for k, name in ((0, "all trades"), (1, "excl. largest winner"), (5, "excl. 5 largest winners"),
                    (max(1, int(round(len(t) * 0.05))), "excl. top 5% of trades"),
                    (max(1, int(round(len(t) * 0.10))), "excl. top 10% of trades")):
        sub = s.iloc[k:]
        b = basic(sub)
        rows.append({"sample": label, "variant": name, "trades": b["trades"], "net_R": b["net_R"],
                     "ci95_low": ci_low(sub.r_multiple), "median_R": b["median_R"], "pf": b["pf"],
                     "total_R": b["total_R"], "total_pnl_usdt": b["total_pnl"]})
    top5 = s.head(5)
    share = {"sample": label, "top5_share_of_net_R_%": top5.r_multiple.sum() / t.r_multiple.sum() * 100,
             "top5_share_of_net_pnl_%": top5.pnl.sum() / t.pnl.sum() * 100,
             "top5_share_of_gross_profit_%": top5.pnl.sum() / t.loc[t.pnl > 0, "pnl"].sum() * 100,
             "top5_trades": "; ".join(f"{r.pair[:-4]} {r.entry_ts:%Y-%m} {r.r_multiple:+.1f}R" for r in top5.itertuples())}
    return rows, share


# ------------------------------------------------------------------ 2) leave one coin out
def loco(t):
    rows = []
    for pair in sorted(t.pair.unique()):
        sub = t[t.pair != pair]
        b, own = basic(sub), basic(t[t.pair == pair])
        rows.append({"left_out": pair, "remaining_trades": b["trades"], "remaining_net_R": b["net_R"],
                     "remaining_ci_low": ci_low(sub.r_multiple), "remaining_pf": b["pf"],
                     "coin_trades": own["trades"], "coin_net_R": own["net_R"], "coin_pf": own["pf"],
                     "coin_total_R": own["total_R"]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 3) market state at entry
def btc_state(btc):
    det = detect(btc)
    lab = det["trend"].map({"strong_up": "bull", "weak_up": "bull", "range": "sideways",
                            "weak_down": "bear", "strong_down": "bear"})
    return pd.Series(lab.to_numpy(), index=pd.to_datetime(btc["timestamp"], utc=True))


def attach_state(t, btc_lab):
    # the decision was taken at the previous day's close; the label at that close is what was known
    dec = t["entry_ts"] - pd.Timedelta(days=1)
    t = t.copy()
    t["btc_market"] = btc_lab.reindex(dec).fillna("n/a").to_numpy()
    return t


def group(t, by):
    rows = []
    for k, sub in t.groupby(by):
        b = basic(sub)
        rows.append({by: k, "trades": b["trades"], "net_R": b["net_R"], "median_R": b["median_R"],
                     "ci95_low": ci_low(sub.r_multiple) if len(sub) > 1 else np.nan, "pf": b["pf"],
                     "win_pct": b["win_pct"], "total_R": b["total_R"]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 5) matched random benchmark
def random_benchmark(data, ctx, c, start, end, target, n=N_RANDOM, seed=SEED):
    """Per-coin entry probability calibrated so the random trade count matches T50's per coin.

    Random entries are Bernoulli draws from a seeded generator, independent of prices; the entry is
    acted on at the next open; exits are T50's causal 25-day-low exit and 3-ATR stop; sizing, costs and
    the cost filter are the engine's. Nothing uses data after the decision bar.
    """
    segs = {p: V.segment(df, start, end) for p, df in data.items()}
    segs = {p: s for p, s in segs.items() if (pd.to_datetime(s["timestamp"], utc=True) >= pd.Timestamp(start, tz="UTC")).sum() >= 50}
    sigs = {p: R1.signals("T50", s, {**ctx, "pair": p}) for p, s in segs.items()}

    def one(pair, prob, rng):
        sig = sigs[pair].copy()
        sig["entry"] = rng.random(len(sig)) < prob
        cc = copy.deepcopy(c)
        cc.strategy, cc.trade_from = "trend_following", start
        return run_backtest(segs[pair], cc, pair, signals=sig)

    probs = {}
    for pair in segs:                                       # calibrate (3 rounds of 30 pilot runs, own seeds)
        p = max(target.get(pair, 1), 1) / max(len(segs[pair]), 1) * 3
        for it in range(3):
            rng = np.random.default_rng(10_000 + it)
            got = np.mean([len(one(pair, p, rng).trades) for _ in range(30)])
            p = float(np.clip(p * max(target.get(pair, 1), 1) / max(got, 0.5), 1e-4, 0.9))
        probs[pair] = p
    rng = np.random.default_rng(seed)
    runs = []
    for _ in range(n):
        res = {pair: one(pair, probs[pair], rng) for pair in segs}
        t = pd.concat([r.trades for r in res.values() if len(r.trades)], ignore_index=True) \
            if any(len(r.trades) for r in res.values()) else pd.DataFrame(columns=["r_multiple", "pnl"])
        gl = -t.loc[t.pnl <= 0, "pnl"].sum() if len(t) else 0
        runs.append({"net_R": t.r_multiple.mean() if len(t) else np.nan, "trades": len(t),
                     "pf": t.loc[t.pnl > 0, "pnl"].sum() / gl if gl > 0 else np.nan,
                     "total_pnl": t.pnl.sum() if len(t) else 0.0,
                     "exposure_pct": np.mean([r.equity["in_position"].mean() for r in res.values()]) * 100})
    return pd.DataFrame(runs), probs


def describe_dist(real, runs, label):
    q = runs["net_R"].quantile([0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {"sample": label, "T50_net_R": real["net_R"], "T50_trades": real["trades"],
            "T50_exposure_pct": real["exposure_pct"], "random_runs": len(runs),
            "random_mean_R": runs["net_R"].mean(), "random_p5": q[0.05], "random_median": q[0.5],
            "random_p95": q[0.95], "random_p99": q[0.99], "random_max": runs["net_R"].max(),
            "random_mean_trades": runs["trades"].mean(), "random_mean_exposure_pct": runs["exposure_pct"].mean(),
            "T50_percentile": (runs["net_R"] < real["net_R"]).mean() * 100,
            "empirical_p_value": ((runs["net_R"] >= real["net_R"]).sum() + 1) / (len(runs) + 1),
            "T50_pnl": real["pnl"], "random_mean_pnl": runs["total_pnl"].mean(),
            "pnl_percentile": (runs["total_pnl"] < real["pnl"]).mean() * 100}


# ------------------------------------------------------------------ portfolio inputs
def portfolio_inputs(data, ctx):
    frames, ent, ex, atr = {}, {}, {}, {}
    for pair, df in data.items():
        sig = R1.signals("T50", df, {**ctx, "pair": pair})
        idx = pd.to_datetime(df["timestamp"], utc=True)
        frames[pair] = df.set_index(idx)[["Open", "High", "Low", "Close", "Volume"]]
        ent[pair] = pd.Series(sig["entry"].to_numpy(bool), index=idx)
        ex[pair] = pd.Series(sig["exit"].to_numpy(bool), index=idx)
        atr[pair] = pd.Series(sig["ATR"].to_numpy(float), index=idx)
    return frames, ent, ex, atr


def conservative():
    return RiskLimits(risk_per_trade=0.005, max_position_pct=0.10, max_exposure_pct=0.30, max_open_positions=3,
                      max_daily_loss_pct=0.03, max_drawdown_pct=0.10, min_notional=1.0, max_cost_to_risk=0.25,
                      kill_switch_file="")


def passive(frames, start, end, weight, capital):
    """Equal-weight buy-and-hold of the universe, scaled to ``weight`` of capital (rest in cash)."""
    closes = pd.DataFrame({p: f["Close"] for p, f in frames.items()})
    closes = closes[(closes.index >= pd.Timestamp(start, tz="UTC")) & (closes.index < pd.Timestamp(end, tz="UTC"))]
    rel = closes / closes.bfill().iloc[0]
    basket = rel.ffill().mean(axis=1)               # coins without data yet count at 1.0 (cash) until listed
    eq = capital * (1 - weight + weight * basket)
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    return {"total_return_pct": (eq.iloc[-1] / capital - 1) * 100,
            "cagr_pct": ((eq.iloc[-1] / capital) ** (1 / years) - 1) * 100,
            "max_dd_pct": -(eq / eq.cummax() - 1).min() * 100}, eq


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    os.makedirs(FIG, exist_ok=True)
    data, hold = R1.load(R1.DEFAULT_UNIVERSE), R1.load(R1.HOLDOUT)
    btc_ok = R1.btc_filter(data["BTCUSDT"])
    member, _ = R1.rs_basket(data, btc_ok)
    ctx, hctx = {"btc_ok": btc_ok, "member": member}, {"btc_ok": btc_ok, "member": member}
    btc_lab = btc_state(data["BTCUSDT"])
    txt = [f"# T50 scrutiny ({pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC)\n",
           "T50 rules frozen as pre-registered: daily close > previous 50-day highest close -> buy next open; "
           "exit at next open after a close < previous 25-day lowest close; 3-ATR(14) protective stop; no "
           "take-profit; no trailing stop. Unchanged engine, costs (0.20%/side fee, 0.05% slippage, 0.20% stop "
           "slippage) and Conservative risk. **Data >= 2025-01-01 never loaded.**\n",
           "**Change in measurement vs Round 1 (not in the rules):** positions still open at the end of a period "
           "or at a coin's delisting are now closed at the last available close and counted. Round 1 dropped "
           "them (engine default `close_at_end=False`), which can hide losses. Both versions are shown.\n"]

    c = cfg(True)
    res = {"main 2019-24": R1.run("T50", data, ctx, c, *ALL), "main DEV": R1.run("T50", data, ctx, c, *DEV),
           "main VAL": R1.run("T50", data, ctx, c, *VAL), "holdout 2019-24": R1.run("T50", hold, hctx, c, *ALL)}
    old = {"main DEV": R1.run("T50", data, ctx, cfg(False), *DEV), "main VAL": R1.run("T50", data, ctx, cfg(False), *VAL),
           "holdout 2019-24": R1.run("T50", hold, hctx, cfg(False), *ALL)}
    T = {k: attach_state(trades_of(v), btc_lab) for k, v in res.items()}
    T["all 25 coins 2019-24"] = pd.concat([T["main 2019-24"], T["holdout 2019-24"]], ignore_index=True)
    cmp_rows = []
    for k in old:
        a, b = basic(trades_of(old[k])), basic(T[k])
        cmp_rows.append({"sample": k, "round1_trades": a["trades"], "round1_net_R": a["net_R"],
                         "now_trades": b["trades"], "now_net_R": b["net_R"], "now_ci_low": ci_low(T[k].r_multiple),
                         "now_pf": b["pf"]})
    txt += ["## 0. Effect of counting positions open at period end / delisting\n", md(pd.DataFrame(cmp_rows)), "\n"]
    print("base done", round(time.time() - t0), flush=True)

    # 1) outliers
    rows, shares = [], []
    for k in ("main DEV", "main VAL", "main 2019-24", "holdout 2019-24", "all 25 coins 2019-24"):
        r, s = outliers(T[k], k)
        rows += r
        shares.append(s)
    txt += ["## 1. Outlier sensitivity\n", md(pd.DataFrame(rows)), "\n", "Concentration of profit in the 5 largest trades:\n",
            md(pd.DataFrame(shares)), "\n"]

    # 2) leave-one-coin-out
    lo_main, lo_all = loco(T["main 2019-24"]), loco(T["all 25 coins 2019-24"])
    txt += ["## 2. Coin independence\n", "### Main universe 2019-2024: leave one coin out (and that coin alone)\n",
            md(lo_main), "\n", "### All 25 coins 2019-2024: leave one coin out\n", md(lo_all), "\n"]

    # 3) time and regime
    tm = T["all 25 coins 2019-24"]
    yr_main, yr_hold, yr_all = group(T["main 2019-24"], "year"), group(T["holdout 2019-24"], "year"), group(tm, "year")
    mk = group(tm, "btc_market")
    rg = group(tm.assign(coin_regime=tm["regime"].fillna("n/a")), "coin_regime")
    ex2021 = tm[~tm.year.isin([2020, 2021])]
    b = basic(ex2021)
    out_bull = pd.DataFrame([{"sample": "all 25 coins, excluding 2020 and 2021 entries", "trades": b["trades"],
                              "net_R": b["net_R"], "ci95_low": ci_low(ex2021.r_multiple), "median_R": b["median_R"],
                              "pf": b["pf"], "total_R": b["total_R"]}])
    for k, sub in (("main only, excl. 2020-21", T["main 2019-24"][~T["main 2019-24"].year.isin([2020, 2021])]),
                   ("holdout only, excl. 2020-21", T["holdout 2019-24"][~T["holdout 2019-24"].year.isin([2020, 2021])])):
        bb = basic(sub)
        out_bull.loc[len(out_bull)] = [k, bb["trades"], bb["net_R"], ci_low(sub.r_multiple), bb["median_R"], bb["pf"],
                                       bb["total_R"]]
    y22 = tm[tm.year == 2022].sort_values("entry_ts")
    y22_tab = y22[["pair", "entry_time", "exit_time", "bars_held", "exit_reason", "r_multiple", "btc_market"]]
    txt += ["## 3. Time-period robustness\n", "### By entry year\n", "Main universe:\n", md(yr_main), "\nHoldout:\n",
            md(yr_hold), "\nAll 25 coins:\n", md(yr_all), "\n",
            "### By BTC market state at the decision close (OF2 regime detector `core/regime.detect` on BTC daily: "
            "strong/weak up = bull, range = sideways, weak/strong down = bear)\n", md(mk), "\n",
            "### By the coin's own engine regime label at entry (`classify_regime`)\n", md(rg), "\n",
            "### Outside the 2020-21 bull market\n", md(out_bull), "\n",
            "### Every 2022 trade (all 25 coins)\n", md(y22_tab.reset_index(drop=True)), "\n"]
    print("sections 1-3 done", round(time.time() - t0), flush=True)

    # 4) drawdown reconciliation (main DEV, per-coin engine accounts)
    dev = res["main DEV"]
    rows = []
    for pair, r in dev.items():
        e = r.equity.copy()
        e["pos_value"] = e["equity"] - e["cash"]
        e["notional_pct"] = e["pos_value"] / e["equity"] * 100
        dd = (e["equity_low"] / e["equity"].cummax() - 1).min() * 100
        tr = r.trades
        rows.append({"pair": pair, "max_dd_pct (engine metric)": r.metrics["max_drawdown_pct"],
                     "worst_close_to_close_dd_pct": -(e["equity"] / e["equity"].cummax() - 1).min() * 100,
                     "max_position_notional_%_of_account": e["notional_pct"].max(),
                     "worst_realised_trade_R": tr["r_multiple"].min() if len(tr) else np.nan})
    dd_tab = pd.DataFrame(rows)
    bnb = dev["BNBUSDT"]
    e = bnb.equity.copy()
    e["t"] = pd.to_datetime(e["time"], utc=True)
    e["pos_value"] = e["equity"] - e["cash"]
    peak_i = int(e["equity"].idxmax())
    trough_i = int((e["equity_low"] / e["equity"].cummax()).idxmin())
    big = bnb.trades.sort_values("r_multiple").iloc[-1]
    entry_notional = big["qty"] * big["entry_price"]
    acct_at_entry = float(e.loc[e["t"] == pd.Timestamp(big["entry_time"]), "equity"].iloc[0])
    bnb_note = (f"BNB's largest trade: entered {big['entry_time'][:10]} at {big['entry_price']:.2f} with {big['qty']:.3f} BNB "
                f"= {entry_notional:.1f} USDT, i.e. {entry_notional / acct_at_entry * 100:.1f}% of the {acct_at_entry:.0f} USDT "
                f"account. Stop {big['stop_loss']:.2f} -> planned risk {big['risk_amount']:.2f} USDT "
                f"({big['risk_amount'] / acct_at_entry * 100:.2f}% of the account, the 0.5% limit). "
                f"Exited {big['exit_time'][:10]} at {big['exit_price']:.2f} (+{big['r_multiple']:.1f}R, "
                f"{big['pnl']:.0f} USDT). At the account peak ({e.loc[peak_i, 't']:%Y-%m-%d}) the open position was worth "
                f"{e.loc[peak_i, 'pos_value']:.0f} USDT = {e.loc[peak_i, 'pos_value'] / e.loc[peak_i, 'equity'] * 100:.0f}% "
                f"of account equity {e.loc[peak_i, 'equity']:.0f}. The drawdown low ({e.loc[trough_i, 't']:%Y-%m-%d}, "
                f"intraday) was {e.loc[trough_i, 'equity_low']:.0f} USDT.")
    # concurrency / aggregate exposure across the 9 independent accounts
    flags = pd.DataFrame({p: r.equity.set_index(pd.to_datetime(r.equity["time"], utc=True))["in_position"]
                          for p, r in res["main 2019-24"].items()}).fillna(False)
    vals = pd.DataFrame({p: (r.equity["equity"] - r.equity["cash"]).set_axis(pd.to_datetime(r.equity["time"], utc=True))
                         for p, r in res["main 2019-24"].items()}).fillna(0)
    eqs = pd.DataFrame({p: r.equity["equity"].set_axis(pd.to_datetime(r.equity["time"], utc=True))
                        for p, r in res["main 2019-24"].items()}).ffill().bfill()
    conc = flags.sum(axis=1)
    agg_exp = vals.sum(axis=1) / eqs.sum(axis=1) * 100
    conc_tab = conc.value_counts(normalize=True).sort_index().mul(100).rename("share_of_days_%").to_frame()
    txt += ["## 4. Drawdown investigation\n",
            "Round 1 ran **one independent 1,000 USDT account per coin** (the engine is single-asset), compounding "
            "within each account. The 0.5% limit applies to the loss *at the stop at entry*. It does not cap later "
            "drawdowns of open profit.\n",
            md(dd_tab), "\n", bnb_note + "\n",
            "**Reconciliation:** the 22.5% figure is real arithmetic, not a bug. The stop protects only the entry "
            "risk. As BNB rose about 10x, a position that was ~3% of the account became the majority of it. Giving "
            "back part of that open profit (the 25-day-low exit sits far below the peak) is a drawdown of the "
            "account, not a loss of capital. The engine's metric uses intraday lows (`equity_low`). It is a "
            "**concentration effect inside a single-coin account.** A single-account portfolio (section 7) "
            "enforces the 10% per-position cap only at entry, so the same effect can appear there. See its max "
            "drawdown.\n",
            "### Position overlap across the 9 per-coin accounts (2019-2024)\n", md(conc_tab, idx=True), "\n",
            f"Aggregate open notional / total equity of the 9 accounts: mean {agg_exp.mean():.1f}%, max {agg_exp.max():.1f}%. "
            f"Up to {int(conc.max())} coins in position at once. A single Conservative account allows at most 3 "
            "positions and 30% exposure, so Round 1's per-coin figures overstate what one account could hold. "
            "Section 7 is the realistic figure.\n"]
    print("section 4 done", round(time.time() - t0), flush=True)

    # 6) holdout per coin with listing / delisting
    hrows = []
    for pair, df in hold.items():
        ts = pd.to_datetime(df["timestamp"], utc=True)
        tr = T["holdout 2019-24"][T["holdout 2019-24"].pair == pair]
        b = basic(tr)
        hrows.append({"pair": pair, "first_day": f"{ts.min():%Y-%m-%d}", "last_day": f"{ts.max():%Y-%m-%d}",
                      "delisted_before_2025": ts.max() < pd.Timestamp("2024-12-31", tz="UTC"),
                      "buy_hold_%": (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100, "trades": b["trades"],
                      "net_R": b["net_R"], "pf": b["pf"], "total_R": b["total_R"],
                      "closed_at_data_end": int((tr["exit_reason"] == "end_of_data").sum())})
    htab = pd.DataFrame(hrows)
    hb = basic(T["holdout 2019-24"])
    h_elig = htab[htab.trades >= 10]
    txt += ["## 6. Holdout (16 coins never used before; rules unchanged)\n", md(htab), "\n",
            f"Collective: {hb['trades']} trades, net {hb['net_R']:+.3f}R (95% lower bound {ci_low(T['holdout 2019-24'].r_multiple):+.3f}R), "
            f"PF {hb['pf']:.2f}, median {hb['median_R']:+.3f}R; positive on {(h_elig.net_R > 0).sum()} of {len(h_elig)} coins "
            "with >= 10 trades. Delisted coins are included up to their last trading day; a position open at "
            "delisting is closed at the last close (real exits around delisting can be worse).\n"]

    # 5) matched random benchmark
    bench_rows = []
    for label, d_, cx in (("main 2019-24", data, ctx), ("holdout 2019-24", hold, hctx)):
        real_t = T[label]
        target = real_t.groupby("pair").size().to_dict()
        runs, probs = random_benchmark(d_, cx, c, *ALL, target)
        real = {"net_R": real_t.r_multiple.mean(), "trades": len(real_t), "pnl": real_t.pnl.sum(),
                "exposure_pct": np.mean([r.equity["in_position"].mean() for r in res[label].values()]) * 100}
        bench_rows.append(describe_dist(real, runs, label))
        runs.to_csv(os.path.join(ROOT, "state", "research", f"t50_random_{label.split()[0]}.csv"), index=False)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 3.2))
            ax.hist(runs["net_R"], bins=40, color="#9aa5b1")
            ax.axvline(real["net_R"], color="#c0392b", lw=2, label=f"T50 {real['net_R']:+.2f}R")
            ax.set_xlabel("pooled net expectancy (R per trade)")
            ax.set_ylabel("random runs")
            ax.set_title(f"{N_RANDOM} matched random-entry runs, {label}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(FIG, f"t50_random_{label.split()[0]}.png"), dpi=110)
            plt.close(fig)
        except Exception as exc:  # figure is optional
            print("figure failed", exc)
        print("bench", label, round(time.time() - t0), flush=True)
    btab = pd.DataFrame(bench_rows)
    txt += ["## 5. Matched random-entry benchmark\n",
            f"{N_RANDOM} independent runs per sample (seed {SEED}). Per coin, the random entry probability is "
            "calibrated (3 x 30 pilot runs, separate seeds) so the expected number of random trades matches T50's "
            "trades on that coin. Everything else is identical: next-open fills, the 25-day-low exit, the 3-ATR stop, "
            "sizing, costs and the cost filter. Entries are drawn from a seeded generator that never sees prices, "
            "so random entries cannot use future information.\n",
            md(btab), "\n", "Figures: `docs/research/fig/t50_random_main.png`, `docs/research/fig/t50_random_holdout.png`.\n"]

    # 7) economics: single account, Conservative limits across coins
    erows, curves = [], {}
    cost_grid = [("base 0.20% fee, 0.05% slip, 0.20% stop slip", Costs(0.002, 0.0005, 0.002)),
                 ("Binance-like 0.10% fee", Costs(0.001, 0.0005, 0.002)),
                 ("0.30% fee, 0.10% slip, 0.30% stop slip", Costs(0.003, 0.001, 0.003)),
                 ("0.40% fee, 0.20% slip, 0.50% stop slip", Costs(0.004, 0.002, 0.005))]
    for label, d_, cx, per in (("main 9 coins", data, ctx, ALL), ("main 9 coins DEV", data, ctx, DEV),
                               ("main 9 coins VAL", data, ctx, VAL), ("holdout 16 coins", hold, hctx, ALL)):
        fr, en, ex, at = portfolio_inputs(d_, cx)
        cap = 10_000.0
        variants = [(cn, co, conservative()) for cn, co in
                    (cost_grid if label in ("main 9 coins", "holdout 16 coins") else cost_grid[:1])]
        info = conservative()
        info.max_drawdown_pct = 1.0                     # information only: the configured limit is NOT changed
        variants.append(("base costs, INFO ONLY: no 10% drawdown halt", cost_grid[0][1], info))
        for cname, cost, lim in variants:
            r = simulate(fr, en, ex, at, *per, lim, cost, capital=cap)
            s = stats(r, cap)
            pas, peq = passive(fr, *per, s["avg_exposure_pct"] / 100, cap)
            full, feq = passive(fr, *per, 1.0, cap)
            erows.append({"portfolio": label, "costs": cname, **s,
                          "passive_same_exposure_return_%": pas["total_return_pct"],
                          "passive_same_exposure_max_dd_%": pas["max_dd_pct"],
                          "buy_hold_return_%": full["total_return_pct"], "buy_hold_max_dd_%": full["max_dd_pct"]})
            if cname.startswith("base 0.20") and label in ("main 9 coins", "holdout 16 coins"):
                curves[label] = (r.equity, peq, feq, r.exposure)
    etab = pd.DataFrame(erows)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex="col")
        for j, (label, (eq, peq, feq, exp)) in enumerate(curves.items()):
            ax = axes[0, j]
            ax.plot(eq.index, eq / eq.iloc[0], label="T50 single account", color="#c0392b")
            ax.plot(peq.index, peq / peq.iloc[0], label="passive basket at same avg exposure", color="#2471a3")
            ax.plot(feq.index, feq / feq.iloc[0], label="buy & hold basket (100%)", color="#9aa5b1", lw=0.8)
            ax.set_yscale("log")
            ax.set_title(label)
            ax.legend(fontsize=7)
            axes[1, j].fill_between(exp.index, exp * 100, color="#c0392b", alpha=0.4)
            axes[1, j].set_ylabel("exposure % of equity")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "t50_portfolio.png"), dpi=110)
        plt.close(fig)
    except Exception as exc:
        print("figure failed", exc)
    # consistency check: one coin, cross-coin limits irrelevant -> must match the engine closely
    fr, en, ex, at = portfolio_inputs({"BTCUSDT": data["BTCUSDT"]}, ctx)
    one = simulate(fr, en, ex, at, *DEV, RiskLimits(risk_per_trade=0.005, max_position_pct=0.10, max_exposure_pct=0.30,
                                                     max_open_positions=3, max_daily_loss_pct=1.0, max_drawdown_pct=1.0,
                                                     min_notional=1.0, max_cost_to_risk=0.25, kill_switch_file=""),
                   Costs(), capital=1000.0)
    eng = res["main DEV"]["BTCUSDT"].trades
    check = (f"Consistency check (BTC, DEV, single coin, halts off): portfolio simulator {len(one.trades)} trades, "
             f"mean {one.trades.r.mean():+.3f}R, final equity {one.equity.iloc[-1]:.1f}; engine {len(eng)} trades, "
             f"mean {eng.r_multiple.mean():+.3f}R, final equity {res['main DEV']['BTCUSDT'].equity['equity'].iloc[-1]:.1f}.")
    txt += ["## 7. Economic viability: one account, Conservative limits enforced across coins\n",
            "**The configured 10% drawdown halt is permanent (kill switch); when it triggers, no new trades are "
            "taken for the rest of the period.** The `INFO ONLY` rows switch it off to show the rule's own economics; "
            "they are not a proposal to change the limit.\n",
            "10,000 USDT, at most 3 positions, 30% total exposure, 10% per position, 0.5% risk per trade, 3% daily "
            "loss limit, 10% drawdown halt (permanent, as the kill switch). Signals taken in fixed universe order "
            "when slots are short. `passive_same_exposure` = equal-weight buy-and-hold basket holding T50's "
            "average exposure, rest in cash.\n",
            md(etab), "\n", check + "\n", "Figure: `docs/research/fig/t50_portfolio.png`.\n"]
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(txt))
    print(f"wrote {OUT} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
