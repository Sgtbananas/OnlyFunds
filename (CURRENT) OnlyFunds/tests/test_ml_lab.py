"""ML Round 1 research engine: causality, purging, reproducible inference, statistics, register, variants.

Offline, synthetic data only. Nothing here can place an order: the modules are checked for order paths too.
"""
import json
import os
import re
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "research"))
import hypotheses as H  # noqa: E402
import ml_lab as L  # noqa: E402
import portfolio as P  # noqa: E402
import portfolio_variants as PV  # noqa: E402
from core.risk_manager import RiskLimits  # noqa: E402


def daily(seed, n=700, start="2019-01-01"):
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, periods=n, freq="D", tz="UTC")
    drift = np.where((np.arange(n) // 90) % 2 == 0, 0.004, -0.003)
    close = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.03, n)))
    op = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.003, n))
    hi = np.maximum(op, close) * (1 + np.abs(rng.normal(0, 0.01, n)))
    lo = np.minimum(op, close) * (1 - np.abs(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"timestamp": days, "created_at": (days.astype("int64") // 10**6).astype("int64"),
                         "Open": op, "High": hi, "Low": lo, "Close": close, "Volume": 1e7, "QuoteVolume": 1e9 *
                         (1 + 0.2 * np.sin(np.arange(n) / 9)), "Trades": 1000})


@pytest.fixture(scope="module")
def data():
    return {s: daily(i) for i, s in enumerate(["BTCUSDT", "ETHUSDT", "BNBUSDT", "ETCUSDT"])}


@pytest.fixture(scope="module")
def panel(data):
    return L.build_panel(data)


def test_features_are_causal(data, panel):
    cut = pd.Timestamp("2020-06-01", tz="UTC")
    shocked = {}
    for s, df in data.items():
        d = df.copy()
        m = pd.to_datetime(d["timestamp"], utc=True) >= cut
        d.loc[m, ["Open", "High", "Low", "Close"]] *= np.linspace(0.5, 2.0, m.sum())[:, None]
        d.loc[m, "QuoteVolume"] *= 3
        shocked[s] = d
    p2 = L.build_panel(shocked)
    a = panel[panel.day < cut].set_index(["pair", "day"])[L.FEATURES]
    b = p2[p2.day < cut].set_index(["pair", "day"])[L.FEATURES]
    pd.testing.assert_frame_equal(a, b)


def test_labels_look_forward_only_and_end_when_data_ends(data):
    lab = L.labels(data["ETHUSDT"])
    df = data["ETHUSDT"].set_index(pd.to_datetime(data["ETHUSDT"]["timestamp"], utc=True))
    t = lab.index[300]
    expect = np.log(df["Open"].iloc[300 + L.HORIZON + 1] / df["Open"].iloc[301])
    assert lab["fwd_ret"].iloc[300] == pytest.approx(expect)
    assert lab["label_end"].iloc[300] == t + pd.Timedelta(days=L.HORIZON + 1)
    assert lab[["y_up", "y_stress", "y_vol"]].iloc[-(L.HORIZON + 1):].isna().all().all()


def test_fold_purge_and_embargo(panel):
    for year in (2019, 2020):
        tr, te = L.fold_masks(panel["day"], panel["label_end"], year)
        start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        assert (panel.loc[tr, "label_end"] < start - pd.Timedelta(days=L.EMBARGO_DAYS)).all()
        assert (panel.loc[te, "day"].dt.year == year).all()
        assert not (tr & te).any()


def test_predict_never_invents_values(panel):
    tr = panel[L.FEATURES].notna().all(axis=1) & panel["y_up"].notna()
    m, _ = L.fit("logit", panel.loc[tr, L.FEATURES], panel.loc[tr, "y_up"])
    X = panel[L.FEATURES].head(400)
    p = L.predict("logit", m, X)
    assert p[X.isna().any(axis=1)].isna().all()                       # missing features -> NaN, no default
    assert p[X.notna().all(axis=1)].between(0, 1).all()


def test_models_are_deterministic(panel):
    tr = panel[L.FEATURES].notna().all(axis=1) & panel["y_up"].notna()
    X, y = panel.loc[tr, L.FEATURES], panel.loc[tr, "y_up"]
    a, _ = L.fit("gbm", X, y)
    b, _ = L.fit("gbm", X, y)
    np.testing.assert_array_equal(a.predict_proba(X.to_numpy(float)), b.predict_proba(X.to_numpy(float)))


def test_card_inference_refuses_on_any_change(tmp_path, data, panel, monkeypatch):
    monkeypatch.setattr(L, "ROOT", str(tmp_path))
    tr = panel[L.FEATURES].notna().all(axis=1) & panel["y_up"].notna()
    m, n = L.fit("logit", panel.loc[tr, L.FEATURES], panel.loc[tr, "y_up"])
    path = os.path.join(str(tmp_path), "m.pkl")
    joblib.dump(m, path)
    dh = L.data_hashes(data)
    card = L.model_card("M1", 2020, "logit", "y_up", L.FEATURES, ("2019-07-01", "2019-12-26"), n, dh, path)
    X = panel[L.FEATURES].tail(50)
    assert L.predict_with_card(card, X, dh).notna().any()
    with pytest.raises(RuntimeError, match="feature list"):
        L.predict_with_card(card, X[L.FEATURES[::-1]], dh)
    with pytest.raises(RuntimeError, match="data hashes"):
        L.predict_with_card(card, X, {**dh, "BTCUSDT": "0" * 64})
    with open(path, "ab") as f:
        f.write(b"x")
    with pytest.raises(RuntimeError, match="model file"):
        L.predict_with_card(card, X, dh)


def test_data_hash_changes_with_any_candle(data):
    d2 = {k: v.copy() for k, v in data.items()}
    d2["ETHUSDT"].loc[100, "Close"] *= 1.0001
    assert L.data_hashes(d2)["ETHUSDT"] != L.data_hashes(data)["ETHUSDT"]
    assert L.data_hashes(d2)["BTCUSDT"] == L.data_hashes(data)["BTCUSDT"]


def test_holm_and_block_bootstrap():
    adj = L.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})
    v = np.r_[np.ones(50), -np.ones(50)] * 0.1 + 0.05
    blocks = np.repeat(np.arange(10), 10)
    m, lo, hi, p = L.block_bootstrap_mean(v, blocks, n=2000)
    assert lo <= m <= hi and m == pytest.approx(0.05)
    assert L.block_bootstrap_mean(v, blocks, n=2000) == (m, lo, hi, p)


def test_register_is_append_only_and_tamper_evident(tmp_path):
    reg = str(tmp_path / "h.jsonl")
    H.register("X1", "s", "ai", "doc", "rule", ("2019-01-01", "2024-12-31"), path=reg)
    with pytest.raises(ValueError, match="already registered"):
        H.register("X1", "s2", "ai", "doc", "rule", ("2019-01-01", "2024-12-31"), path=reg)
    with pytest.raises(ValueError, match="consumed test window"):
        H.register("X2", "s", "human", "doc", "rule", ("2019-01-01", "2025-06-30"), path=reg)
    with pytest.raises(ValueError, match="never registered"):
        H.record_result("X9", "supported", "x", "ref", path=reg)
    H.record_result("X1", "rejected", "no edge", "ref", path=reg)
    assert H.verify(reg) == []
    lines = open(reg, encoding="utf-8").read().splitlines()
    lines[0] = lines[0].replace('"statement": "s"', '"statement": "edited"')
    open(reg, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    assert any("chain broken" in p for p in H.verify(reg))


def test_project_register_chain_is_intact():
    assert H.verify() == []
    ids = [e["id"] for e in H.entries() if e["kind"] == "hypothesis"]
    assert {"H-ML1", "H-ML6", "H-G3", "H-R3"} <= set(ids)


def _inputs(data):
    frames, ent, ex, atr = {}, {}, {}, {}
    for s, df in data.items():
        idx = pd.to_datetime(df["timestamp"], utc=True)
        f = df.set_index(idx)
        c = f["Close"]
        frames[s] = f[["Open", "High", "Low", "Close", "Volume"]]
        ent[s] = c > c.rolling(50).max().shift(1)
        ex[s] = c < c.rolling(25).min().shift(1)
        atr[s] = L.atr14(df)
    return frames, ent, ex, atr


def _limits():
    return RiskLimits(risk_per_trade=0.005, max_position_pct=0.10, max_exposure_pct=0.30, max_open_positions=3,
                      max_daily_loss_pct=1.0, max_drawdown_pct=1.0, min_notional=1.0, max_cost_to_risk=0.25,
                      kill_switch_file="")


def test_variant_simulator_matches_frozen_simulator_when_off(data):
    fr, en, ex, at = _inputs(data)
    a = P.simulate(fr, en, ex, at, "2019-03-01", "2020-12-01", _limits(), capital=10000.0)
    b, risk = PV.simulate(fr, en, ex, at, "2019-03-01", "2020-12-01", _limits(), capital=10000.0)
    np.testing.assert_allclose(a.equity.to_numpy(), b.equity.to_numpy())
    np.testing.assert_allclose(a.trades["pnl"].to_numpy(), b.trades["pnl"].to_numpy())
    assert {"initial_risk_pct", "unrealised_pct", "open_risk_to_stop_pct", "max_position_pct", "drawdown_pct"} <= set(risk)


def test_trim_band_caps_concentration_and_profit_lock_tags_exits(data):
    fr, en, ex, at = _inputs(data)
    base, rb = PV.simulate(fr, en, ex, at, "2019-03-01", "2020-12-01", _limits(), capital=10000.0)
    tb, rt = PV.simulate(fr, en, ex, at, "2019-03-01", "2020-12-01", _limits(), capital=10000.0, trim_band=(0.04, 0.02))
    assert (tb.trades.reason == "trim").any()
    # after a trim the position is cut back: between closes it can only grow by that day's move
    assert rt["max_position_pct"].max() < max(rb["max_position_pct"].max(), 4.0) * 1.5
    pl, _ = PV.simulate(fr, en, ex, at, "2019-03-01", "2020-12-01", _limits(), capital=10000.0, profit_lock=(0.5, 2.0))
    assert set(pl.trades.reason) <= {"signal_exit", "stop_loss", "profit_lock", "end_of_period"}
    assert (pl.trades.reason == "profit_lock").any()


def test_ml_code_cannot_trade_and_is_isolated_from_shadow_tracking():
    for f in ("ml_lab.py", "ml_round1.py", "portfolio_variants.py", "hypotheses.py"):
        src = open(os.path.join(ROOT, "research", f), encoding="utf-8").read()
        for bad in ("place_order", "cancel_order", "CoinExClient", "API_SECRET", "allow_trading", "requests.post",
                    "LIVE_TRADING_LOCKED =", "import forward_shadow", "OF2_STATE_DIR\") or os.path.join(ROOT, \"state\"), \"shadow"):
            assert bad not in src, (f, bad)
        assert not re.search(r"""["']shadow["']""", src), f            # never builds a path into the shadow state
    shadow = open(os.path.join(ROOT, "research", "forward_shadow.py"), encoding="utf-8").read()
    for mod in ("ml_lab", "ml_round1", "portfolio_variants", "hypotheses"):
        assert mod not in shadow


def test_frozen_shadow_fingerprints_still_match():
    import forward_shadow as FS
    _, problems = FS.load_frozen()
    assert problems == []


def test_fake_sentiment_placeholder_is_never_imported():
    import glob
    hits = []
    for f in glob.glob(os.path.join(ROOT, "**", "*.py"), recursive=True):
        if os.sep + "legacy" + os.sep in f or f.endswith("test_ml_lab.py"):
            continue
        src = open(f, encoding="utf-8", errors="ignore").read()
        if re.search(r"(from|import)\s+legacy(\.sentiment)?\b|fetch_sentiment_score", src):
            hits.append(f)
    assert hits == []
