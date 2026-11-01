"""Experimental market-regime engine: probabilities with confidence, walk-forward, against simple baselines.

Research only. It reads causally available daily market data (Binance 1d, 2019-2024; the consumed 2025-26 window is
never loaded) and outputs, for each day's close, the probability of each regime over the NEXT 20 days:

- direction:  bull (BTC 20-day forward log return > +5%), bear (< -5%), sideways (otherwise)
- vol_high / vol_low: BTC's forward 20-day realised volatility above the 2/3 (below the 1/3) quantile of its own
  trailing 365-day distribution of 20-day volatility, measured at the decision day (a causal threshold)
- vol_expand: forward 20-day volatility > 1.25 x the current 20-day volatility
- stress: BTC's lowest low over the next 20 days is more than 15% below the next open

Features (all known at the close): BTC 5/20/60-day returns, distance to the 200-day average, above the 100-day
average, 20-day volatility, volatility ratio 20/100, ATR expansion, 90-day drawdown, the existing rule trend score,
volume trend, plus breadth (share of the market universe above its 50-day average; recent 50-day breakouts) and
cross-asset correlation (mean 60-day correlation with BTC). The market universe is the main 9 coins, fixed.

Models: L2 logistic regression on standardised features (multinomial for direction), fixed C, no tuning.
Baselines, both fitted on the same training rows:
- climatology: the training base rate;
- rule: a one-variable conditional frequency, P(target | simple state), with the state BTC above / below its
  100-day average (direction, stress) or the current volatility tercile (volatility targets).
Walk-forward: test years 2021-2024, training on decisions whose label ended >= 5 days before the test year.
Confidence: 1 - entropy / log(K) of the predicted distribution (1 = certain, 0 = uniform).
"""
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import ml_lab as L  # noqa: E402

H = 20
EMBARGO = 5
TEST_YEARS = (2021, 2022, 2023, 2024)
C = 0.1
MIN_TRAIN = 150
TARGETS = {"direction": ["bear", "sideways", "bull"], "vol_high": [0, 1], "vol_low": [0, 1], "vol_expand": [0, 1],
           "stress": [0, 1]}
FEATURES = ["btc_ret_5", "btc_ret_20", "btc_ret_60", "btc_ma200_dist", "btc_above_ma100", "btc_vol_20",
            "btc_vol_ratio", "btc_vol_expansion", "btc_dd_90", "btc_trend_score", "btc_volume_ratio",
            "breadth_ma50", "breadth_brk", "mkt_corr"]


def frame(data: Dict[str, pd.DataFrame], market=L.MAIN) -> pd.DataFrame:
    """One row per day: causal features, the simple states and forward labels (labels never used as features)."""
    mk = {s: data[s] for s in market if s in data}
    btc = L._indexed(data["BTCUSDT"])
    btc_ret = np.log(btc["Close"] / btc["Close"].shift(1))
    cf = {s: L.coin_features(df, L.atr14(df), btc_ret) for s, df in mk.items()}
    m = L.market_features(mk, cf)
    b = cf["BTCUSDT"]
    m["btc_ret_5"] = b["ret_5"]
    m["btc_vol_ratio"] = b["vol_ratio"]
    m["btc_vol_expansion"] = b["vol_expansion"]
    m["btc_volume_ratio"] = b["volume_ratio"]
    o, lo, c = btc["Open"], btc["Low"], btc["Close"]
    lr = np.log(c / c.shift(1))
    vol20 = lr.rolling(H).std()
    q33 = vol20.rolling(365, min_periods=200).quantile(1 / 3)
    q67 = vol20.rolling(365, min_periods=200).quantile(2 / 3)
    m["state_btc_up"] = (c > c.rolling(100).mean()).astype(float).where(c.rolling(100).mean().notna())
    m["state_vol_tercile"] = np.where(vol20 > q67, 2, np.where(vol20 < q33, 0, 1)).astype(float)
    m.loc[q33.isna() | vol20.isna(), "state_vol_tercile"] = np.nan
    fwd = np.log(o.shift(-(H + 1)) / o.shift(-1))
    fvol = lr.rolling(H).std().shift(-H)               # returns of days t+1 .. t+20
    known = fwd.notna() & fvol.notna()
    m["y_direction"] = pd.Series(np.where(fwd > 0.05, 2, np.where(fwd < -0.05, 0, 1)), index=m.index).where(known)
    m["y_vol_high"] = (fvol > q67).astype(float).where(known & q67.notna())
    m["y_vol_low"] = (fvol < q33).astype(float).where(known & q33.notna())
    m["y_vol_expand"] = (fvol > 1.25 * vol20).astype(float).where(known & vol20.notna())
    mn = lo.rolling(H).min().shift(-H) / o.shift(-1) - 1
    m["y_stress"] = (mn < -0.15).astype(float).where(known)
    m["label_end"] = m.index + pd.Timedelta(days=H + 1)
    num = m.columns.drop("label_end")
    m[num] = m[num].astype(float).replace([np.inf, -np.inf], np.nan)
    return m


def rule_state(target):
    return "state_btc_up" if target in ("direction", "stress") else "state_vol_tercile"


def _model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))


def _probs_from_freq(train_y, k, prior=1.0):
    cnt = np.array([(train_y == j).sum() for j in range(k)], float) + prior
    return cnt / cnt.sum()


def walk_forward(m: pd.DataFrame) -> pd.DataFrame:
    """Out-of-sample probabilities for every target and test day: model, climatology and rule baseline."""
    rows = []
    for target, classes in TARGETS.items():
        k = len(classes)
        y = m[f"y_{target}"]
        state = m[rule_state(target)]
        for year in TEST_YEARS:
            start, end = pd.Timestamp(f"{year}-01-01", tz="UTC"), pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
            tr = (m.label_end < start - pd.Timedelta(days=EMBARGO)) & y.notna() & m[FEATURES].notna().all(axis=1) \
                & state.notna()
            te = (m.index >= start) & (m.index < end) & y.notna() & m[FEATURES].notna().all(axis=1) & state.notna()
            if tr.sum() < MIN_TRAIN or te.sum() == 0:
                continue
            ytr = y[tr].astype(int).to_numpy()
            if len(np.unique(ytr)) < k:                 # every class must be seen in training
                continue
            mod = _model().fit(m.loc[tr, FEATURES].to_numpy(), ytr)
            pm = mod.predict_proba(m.loc[te, FEATURES].to_numpy())
            clim = _probs_from_freq(ytr, k)
            st_tr, st_te = state[tr].to_numpy(), state[te].to_numpy()
            rule = np.vstack([_probs_from_freq(ytr[st_tr == s], k) for s in st_te])
            out = pd.DataFrame({"day": m.index[te], "target": target, "year": year,
                                "y": y[te].astype(int).to_numpy(), "train_rows": int(tr.sum())})
            for j in range(k):
                out[f"p{j}_model"] = pm[:, j]
                out[f"p{j}_clim"] = clim[j]
                out[f"p{j}_rule"] = rule[:, j]
            rows.append(out)
    return pd.concat(rows, ignore_index=True)


def log_loss(df, src, k):
    p = np.column_stack([df[f"p{j}_{src}"].to_numpy() for j in range(k)])
    p = np.clip(p, 1e-6, 1)
    return -np.log(p[np.arange(len(df)), df.y.to_numpy()])


def confidence(df, src, k):
    p = np.clip(np.column_stack([df[f"p{j}_{src}"].to_numpy() for j in range(k)]), 1e-12, 1)
    return 1 - (-(p * np.log(p)).sum(axis=1)) / np.log(k)


def month_block_ci(diff: np.ndarray, days: pd.Series, n=2000, seed=707):
    """90% interval of the mean of `diff` resampling whole calendar months (labels overlap within a month)."""
    mon = pd.Series(pd.to_datetime(days.to_numpy()).strftime("%Y-%m"))
    groups = [np.flatnonzero(mon.to_numpy() == g) for g in mon.unique()]
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        means.append(diff[idx].mean())
    return float(np.percentile(means, 5)), float(np.percentile(means, 95))


def evaluate(oos: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, classes in TARGETS.items():
        k = len(classes)
        g0 = oos[oos.target == target]
        for span, g in (("2021-24", g0), ("2021-22", g0[g0.year <= 2022]), ("2023-24", g0[g0.year >= 2023])):
            if not len(g):
                continue
            ll = {s: log_loss(g, s, k) for s in ("model", "clim", "rule")}
            best = "clim" if ll["clim"].mean() <= ll["rule"].mean() else "rule"
            diff = ll[best] - ll["model"]                     # > 0: model better than the best baseline
            lo, hi = month_block_ci(diff, g.day)
            rows.append({"target": target, "span": span, "days": len(g),
                         "base_rate_or_mode": float((g.y == k - 1).mean()) if k == 2 else
                         {c: float((g.y == j).mean()) for j, c in enumerate(classes)},
                         "ll_model": ll["model"].mean(), "ll_clim": ll["clim"].mean(), "ll_rule": ll["rule"].mean(),
                         "best_baseline": best, "gain_vs_best": float(diff.mean()), "gain_lo": lo, "gain_hi": hi,
                         "skill_vs_best_pct": float(diff.mean() / ll[best].mean() * 100),
                         "mean_confidence_model": float(confidence(g, "model", k).mean())})
    return pd.DataFrame(rows)


def calibration(oos: pd.DataFrame, target: str, bins=5) -> pd.DataFrame:
    g = oos[oos.target == target]
    q = pd.qcut(g.p1_model, bins, duplicates="drop")
    return g.groupby(q, observed=True).agg(predicted=("p1_model", "mean"), observed=("y", "mean"), days=("y", "size"))


def verdict(ev: pd.DataFrame, target: str) -> str:
    e = ev[ev.target == target].set_index("span")
    if "2021-24" not in e.index:
        return "rejected (no out-of-sample predictions)"
    ok = e.loc["2021-24", "gain_lo"] > 0 and all(e.loc[s, "gain_vs_best"] > 0 for s in ("2021-22", "2023-24") if s in e.index)
    if ok:
        return "supported"
    return "inconclusive" if e.loc["2021-24", "gain_vs_best"] > 0 else "rejected"


def risk_off_model(oos: pd.DataFrame, mult=2.0) -> pd.Series:
    """Model-based risk-off flag by day: P(stress) >= mult x that fold's training base rate (climatology)."""
    g = oos[oos.target == "stress"]
    return pd.Series((g.p1_model >= mult * g.p1_clim).to_numpy(), index=pd.DatetimeIndex(g.day))
