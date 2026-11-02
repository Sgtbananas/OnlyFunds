"""ML research library for ML Round 1 (pre-registered in docs/research/ML_PROGRAMME.md).

RESEARCH ONLY. No orders, no keys, no exchange client. Independent of the frozen T50/RS3 shadow tracking
(nothing here is imported by research/forward_shadow.py, and nothing here touches state/shadow/).

Contents
- Causal features per coin-day (coin + market), labels known only after the decision, purged walk-forward folds.
- Fixed-hyperparameter models (no search), trained per fold.
- Model cards: code hash, feature list, target, training range, data hashes, hyperparameters, model-file hash.
  `predict_with_card` refuses to run when any of them differ.
- Month-block and week-cluster bootstrap helpers.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

MAIN = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "LTCUSDT", "ADAUSDT", "EOSUSDT", "XLMUSDT", "TRXUSDT"]
HOLDOUT = ["ETCUSDT", "LINKUSDT", "NEOUSDT", "QTUMUSDT", "IOTAUSDT", "ONTUSDT", "ICXUSDT", "WAVESUSDT", "BATUSDT",
           "VETUSDT", "ZILUSDT", "NULSUSDT", "ZECUSDT", "DASHUSDT", "XMRUSDT", "OMGUSDT"]
UNIVERSE = MAIN + HOLDOUT
CUTOFF = pd.Timestamp("2025-01-01", tz="UTC")       # consumed test window starts here: never loaded
TEST_YEARS = (2021, 2022, 2023, 2024)
EMBARGO_DAYS = 5
HORIZON = 20                                        # label covers opens t+1 .. t+21
UP_COST = 0.005
STRESS_DD = -0.20
SEED = 7

COIN_FEATURES = ["brk_atr", "exit_dist_atr", "ret_5", "ret_20", "ret_60", "ma200_dist", "ma50_slope", "atr_pct",
                 "vol_ratio", "vol_expansion", "volume_ratio", "trend_persist", "rs_rank", "corr_btc_60"]
MARKET_FEATURES = ["btc_ret_20", "btc_ret_60", "btc_ma200_dist", "btc_above_ma100", "btc_vol_20", "btc_dd_90",
                   "btc_trend_score", "breadth_ma50", "breadth_brk", "mkt_corr"]
FEATURES = COIN_FEATURES + MARKET_FEATURES
TRADE_FEATURES = FEATURES + ["prior_r_shrunk"]

MODEL_SPECS = {
    "M1": ("logit", "y_up"), "M2": ("gbm", "y_up"), "M1s": ("logit", "y_stress"), "M2s": ("gbm", "y_stress"),
    "MV": ("ridge", "y_vol"), "M3": ("logit", "y_win"), "M4": ("gbm_small", "y_win"),
}


# ------------------------------------------------------------------ data
def load_universe(symbols=UNIVERSE) -> Dict[str, pd.DataFrame]:
    import round1 as R1
    data = R1.load(symbols)                          # asserts every candle < 2025-01-01
    for s, df in data.items():
        assert pd.to_datetime(df["timestamp"], utc=True).max() < CUTOFF, s
    return data


def data_hashes(data: Dict[str, pd.DataFrame]) -> Dict[str, str]:
    out = {}
    for s, df in sorted(data.items()):
        cols = df[["timestamp", "Open", "High", "Low", "Close", "Volume"]].copy()
        cols["timestamp"] = pd.to_datetime(cols["timestamp"], utc=True).astype("int64")
        out[s] = hashlib.sha256(pd.util.hash_pandas_object(cols, index=False).values.tobytes()).hexdigest()
    return out


def _indexed(df):
    return df.set_index(pd.to_datetime(df["timestamp"], utc=True))


def atr14(df: pd.DataFrame) -> pd.Series:
    """The engine's own ATR (via round1.base_frame -> core.core_signals), indexed by UTC day."""
    import round1 as R1
    sig = R1.base_frame(df)
    return pd.Series(sig["ATR"].to_numpy(float), index=pd.to_datetime(df["timestamp"], utc=True))


# ------------------------------------------------------------------ features (causal)
def coin_features(df: pd.DataFrame, atr: pd.Series, btc_ret: pd.Series) -> pd.DataFrame:
    d = _indexed(df)
    c, qv = d["Close"], d["QuoteVolume"] if "QuoteVolume" in d else d["Close"] * d["Volume"]
    lr = np.log(c / c.shift(1))
    f = pd.DataFrame(index=d.index)
    f["brk_atr"] = (c - c.rolling(50).max().shift(1)) / atr
    f["exit_dist_atr"] = (c - c.rolling(25).min().shift(1)) / atr
    for n in (5, 20, 60):
        f[f"ret_{n}"] = np.log(c / c.shift(n))
    sma50, sma200 = c.rolling(50).mean(), c.rolling(200).mean()
    f["ma200_dist"] = np.log(c / sma200)
    f["ma50_slope"] = np.log(sma50 / sma50.shift(10))
    f["atr_pct"] = atr / c
    f["vol_ratio"] = lr.rolling(20).std() / lr.rolling(100).std()
    f["vol_expansion"] = atr / atr.shift(20)
    f["volume_ratio"] = np.log(qv.rolling(20).mean() / qv.rolling(100).mean())
    f["trend_persist"] = (c > sma50).astype(float).where(sma50.notna()).rolling(50).mean()
    f["corr_btc_60"] = lr.rolling(60).corr(btc_ret.reindex(d.index))
    return f.replace([np.inf, -np.inf], np.nan)


def market_features(data: Dict[str, pd.DataFrame], coin_f: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    from core.regime import detect
    btc = _indexed(data["BTCUSDT"])
    c = btc["Close"]
    lr = np.log(c / c.shift(1))
    m = pd.DataFrame(index=btc.index)
    m["btc_ret_20"] = np.log(c / c.shift(20))
    m["btc_ret_60"] = np.log(c / c.shift(60))
    m["btc_ma200_dist"] = np.log(c / c.rolling(200).mean())
    m["btc_above_ma100"] = (c > c.rolling(100).mean()).astype(float).where(c.rolling(100).mean().notna())
    m["btc_vol_20"] = lr.rolling(20).std()
    m["btc_dd_90"] = c / c.rolling(90).max() - 1
    m["btc_trend_score"] = detect(data["BTCUSDT"])["trend_score"].to_numpy(float)
    closes = pd.DataFrame({s: _indexed(df)["Close"] for s, df in data.items()})
    above = closes > closes.rolling(50).mean()
    m["breadth_ma50"] = above.where(closes.rolling(50).mean().notna()).mean(axis=1).reindex(m.index)
    brk = closes > closes.rolling(50).max().shift(1)
    m["breadth_brk"] = brk.rolling(5).max().where(closes.rolling(55).count() >= 55).mean(axis=1).reindex(m.index)
    corr = pd.DataFrame({s: f["corr_btc_60"] for s, f in coin_f.items() if s != "BTCUSDT"})
    m["mkt_corr"] = corr.mean(axis=1).reindex(m.index)
    return m


def rs_rank(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    closes = pd.DataFrame({s: _indexed(df)["Close"] for s, df in data.items()})
    mom = closes / closes.shift(28) - 1
    return mom.rank(axis=1, pct=True)


def labels(df: pd.DataFrame) -> pd.DataFrame:
    """Targets for a decision at the close of day t (entry at open t+1). Never used as features."""
    d = _indexed(df)
    o, lo, c = d["Open"], d["Low"], d["Close"]
    lr = np.log(c / c.shift(1))
    out = pd.DataFrame(index=d.index)
    fwd = np.log(o.shift(-(HORIZON + 1)) / o.shift(-1))
    out["fwd_ret"] = fwd
    out["y_up"] = (fwd > UP_COST).astype(float).where(fwd.notna())
    mn = lo.rolling(HORIZON).min().shift(-HORIZON) / o.shift(-1) - 1
    out["y_stress"] = (mn < STRESS_DD).astype(float).where(mn.notna() & fwd.notna())
    out["y_vol"] = np.log(lr.rolling(HORIZON).std().shift(-HORIZON)).where(fwd.notna())
    out["label_end"] = out.index + pd.Timedelta(days=HORIZON + 1)
    out["vol_persist"] = np.log(lr.rolling(HORIZON).std())        # persistence baseline for y_vol (causal)
    return out


def build_panel(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Long panel: one row per coin-day with FEATURES, labels and the BTC rule-regime class (all causal)."""
    from core.regime import detect
    btc = _indexed(data["BTCUSDT"])
    btc_ret = np.log(btc["Close"] / btc["Close"].shift(1))
    atrs = {s: atr14(df) for s, df in data.items()}
    cf = {s: coin_features(df, atrs[s], btc_ret) for s, df in data.items()}
    mk = market_features(data, cf)
    rr = rs_rank(data)
    cls = pd.Series(detect(data["BTCUSDT"])["trend"].to_numpy(), index=btc.index)
    rows = []
    for s, df in data.items():
        f = cf[s].copy()
        f["rs_rank"] = rr[s].reindex(f.index)
        f = f.join(mk, how="left").join(labels(df), how="left")
        f["btc_class"] = cls.reindex(f.index).to_numpy()
        f["pair"] = s
        f["atr"] = atrs[s]
        rows.append(f)
    p = pd.concat(rows)
    p.index.name = "day"
    return p.reset_index()


# ------------------------------------------------------------------ folds
def fold_masks(day: pd.Series, label_end: pd.Series, year: int) -> Tuple[pd.Series, pd.Series]:
    """Train: label fully known EMBARGO_DAYS before the test year. Test: decision day inside the year."""
    start = pd.Timestamp(f"{year}-01-01", tz="UTC")
    end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
    train = label_end < start - pd.Timedelta(days=EMBARGO_DAYS)
    test = (day >= start) & (day < end)
    return train, test


# ------------------------------------------------------------------ models
def make_model(kind: str):
    if kind == "logit":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    if kind == "gbm":
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=200,
                                              l2_regularization=1.0, early_stopping=False, random_state=SEED)
    if kind == "gbm_small":
        return HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05, max_iter=100, min_samples_leaf=20,
                                              l2_regularization=1.0, early_stopping=False, random_state=SEED)
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    raise ValueError(kind)


MIN_TRAIN_ROWS = 50


def fit(kind: str, X: pd.DataFrame, y: pd.Series):
    ok = X.notna().all(axis=1) & y.notna()
    if ok.sum() < MIN_TRAIN_ROWS:
        raise ValueError(f"too few complete training rows ({int(ok.sum())})")
    if kind != "ridge" and y[ok].nunique() < 2:
        raise ValueError("training target has one class")
    m = make_model(kind)
    m.fit(X[ok].to_numpy(float), y[ok].to_numpy(float if kind == "ridge" else int))
    return m, int(ok.sum())


def predict(kind: str, model, X: pd.DataFrame) -> pd.Series:
    """NaN where any feature is missing (never a silent default prediction)."""
    out = pd.Series(np.nan, index=X.index)
    ok = X.notna().all(axis=1)
    if ok.any():
        Z = X[ok].to_numpy(float)
        out[ok] = model.predict(Z) if kind == "ridge" else model.predict_proba(Z)[:, 1]
    return out


# ------------------------------------------------------------------ cards
def code_hash() -> str:
    src = inspect.getsource(sys.modules[__name__]).replace("\r\n", "\n")
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


def model_card(name, fold_year, kind, target, features, train_range, n_train, dhash, model_path):
    with open(model_path, "rb") as f:
        mh = hashlib.sha256(f.read()).hexdigest()
    card = {"model": name, "fold_test_year": fold_year, "kind": kind, "target": target, "features": list(features),
            "hyperparameters": make_model(kind).get_params().__repr__() if kind else None, "seed": SEED,
            "training_label_known_before": train_range[1], "training_first_day": train_range[0],
            "n_train_rows": n_train, "embargo_days": EMBARGO_DAYS, "code_sha256": code_hash(),
            "data_sha256": dhash, "model_file": os.path.relpath(model_path, ROOT).replace("\\", "/"),
            "model_sha256": mh}
    card["card_sha256"] = hashlib.sha256(json.dumps(card, sort_keys=True).encode()).hexdigest()
    return card


def predict_with_card(card: dict, X: pd.DataFrame, dhash: Dict[str, str]) -> pd.Series:
    """Reproducible inference: refuses if code, features, data or model file differ from the card."""
    problems = []
    if card["code_sha256"] != code_hash():
        problems.append("feature/label code changed")
    if list(X.columns) != card["features"]:
        problems.append("feature list differs")
    if {k: v for k, v in dhash.items() if k in card["data_sha256"]} != card["data_sha256"]:
        problems.append("training data hashes differ")
    path = os.path.join(ROOT, card["model_file"])
    with open(path, "rb") as f:
        if hashlib.sha256(f.read()).hexdigest() != card["model_sha256"]:
            problems.append("model file changed")
    if problems:
        raise RuntimeError("inference refused: " + "; ".join(problems))
    return predict(card["kind"], joblib.load(path), X)


# ------------------------------------------------------------------ statistics
def log_loss_rows(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def block_bootstrap_mean(values: np.ndarray, blocks: np.ndarray, n: int = 5000, seed: int = 11):
    """Mean over rows, resampling whole blocks (e.g. calendar months). Returns (mean, lo95, hi95, p_le_0)."""
    values = np.asarray(values, float)
    ub, inv = np.unique(blocks, return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=len(ub))
    cnts = np.bincount(inv, minlength=len(ub)).astype(float)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(ub), size=(n, len(ub)))
    means = sums[draw].sum(axis=1) / cnts[draw].sum(axis=1)
    m = values.mean()
    return float(m), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), float((means <= 0).mean())


def week_cluster_ci(r: np.ndarray, weeks: np.ndarray, n: int = 5000, seed: int = 5):
    if len(r) < 2:
        return float("nan"), float("nan")
    _, lo, hi, _ = block_bootstrap_mean(r, weeks, n, seed)
    return lo, hi


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out
