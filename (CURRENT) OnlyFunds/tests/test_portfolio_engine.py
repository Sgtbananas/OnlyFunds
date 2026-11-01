"""Round 4 portfolio engine and regime engine: equivalence, accounting, slot ranking, replacement, risk-off sizing,
multi-strategy exits, regime causality and walk-forward purging. Offline, synthetic data only."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import portfolio_engine as PE  # noqa: E402
import portfolio_pm as PM  # noqa: E402
from test_ml_lab import _inputs, _limits, daily  # noqa: E402

S, E = "2019-03-01", "2020-12-01"
COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ETCUSDT", "LINKUSDT", "XRPUSDT"]


@pytest.fixture(scope="module")
def raw():
    return {s: daily(i) for i, s in enumerate(COINS)}


@pytest.fixture(scope="module")
def inputs(raw):
    return _inputs(raw)


def run(inputs, **kw):
    fr, en, ex, at = inputs
    rules = kw.pop("rules", PE.Rules())
    return PE.simulate(fr, en, ex, at, S, E, _limits(), capital=10000.0, rules=rules, **kw)


@pytest.mark.parametrize("kw", [{}, {"trim_band": (0.15, 0.10), "profit_lock": (2.0, 6.0)},
                                {"open_risk_cap": (0.004, 0.003), "atr_cap": (0.001, 0.0008)}])
def test_reproduces_round3_simulator_when_engine_features_off(inputs, kw):
    fr, en, ex, at = inputs
    a, _ = PM.simulate(fr, en, ex, at, S, E, _limits(), capital=10000.0, rules=PM.Rules(**kw))
    b, _ = run(inputs, rules=PE.Rules(**kw))
    np.testing.assert_allclose(a.equity.to_numpy(), b.equity.to_numpy())
    pd.testing.assert_frame_equal(a.trades, b.trades)


@pytest.mark.parametrize("kw", [{}, {"replace": (5, 0.5)}, {"risk_off": "half"}])
def test_accounting_identity(inputs, kw):
    if kw.get("risk_off") == "half":
        idx = inputs[0]["BTCUSDT"].index
        kw = {"risk_off": pd.Series(np.arange(len(idx)) % 2 == 0, index=idx)}
    res, _ = run(inputs, **kw)
    assert res.equity.iloc[-1] == pytest.approx(10000.0 + res.trades.pnl.sum(), abs=1e-6)


def test_ranking_decides_which_candidates_get_slots_not_how_many(inputs):
    fr, en, ex, at = inputs
    always = {p: pd.Series(True, index=s.index) for p, s in en.items()}       # every coin signals every day
    never = {p: pd.Series(False, index=s.index) for p, s in ex.items()}
    score = {p: pd.Series(float(i), index=s.index) for i, (p, s) in enumerate(en.items())}
    for sign in (1, -1):
        res, _ = PE.simulate(fr, always, never, at, S, E, _limits(), capital=10000.0,
                             rank={p: sign * s for p, s in score.items()})
        first = res.trades.sort_values("position_id").drop_duplicates("position_id").head(3)
        want = list(fr)[-3:] if sign == 1 else list(fr)[:3]
        assert set(first.pair) == set(want)
        assert res.n_open.max() <= 3
    unranked, _ = PE.simulate(fr, always, never, at, S, E, _limits(), capital=10000.0)
    assert set(unranked.trades.sort_values("position_id").pair.head(3)) == set(list(fr)[:3])   # universe order


def test_replacement_only_removes_stale_positions_and_respects_slots(inputs):
    res, _ = run(inputs, replace=(10, 0.5))
    t = res.trades
    rep = t[t.reason == "replaced"]
    assert len(rep) > 0
    assert (rep.bars >= 10).all()
    # a replaced position was below +0.5R at the decision close, so its exit can only be modestly above it
    assert (rep.r < 3.0).all()
    assert res.n_open.max() <= 3


def test_risk_off_never_increases_size(inputs):
    idx = inputs[0]["BTCUSDT"].index
    off = pd.Series(True, index=idx)
    base, _ = run(inputs)
    half, _ = run(inputs, risk_off=off, risk_off_scale=0.5)
    b = base.trades.groupby("position_id").first()
    h = half.trades.groupby("position_id").first()
    assert h.notional_pct_at_entry.max() <= b.notional_pct_at_entry.max() / 2 + 1e-9
    assert (h.initial_risk / 10000.0).max() <= 0.0025 * 1.6      # equity drifts; never near 0.5%


def test_multi_strategy_positions_exit_on_their_own_strategy(inputs):
    fr, en, ex, at = inputs
    never = {p: pd.Series(False, index=s.index) for p, s in ex.items()}
    alt_entry = {p: (f["Close"] > f["Close"].rolling(20).max().shift(1)) for p, f in fr.items()}
    res, _ = PE.simulate(fr, en, ex, at, S, E, _limits(), capital=10000.0,
                         strategies={"A": (en, ex), "B": (alt_entry, never)})
    t = res.trades
    assert set(t.reason) <= {"signal_exit", "stop_loss", "end_of_period"}
    assert res.equity.iloc[-1] == pytest.approx(10000.0 + t.pnl.sum(), abs=1e-6)


def test_corr_risk_cap_trims(inputs):
    res, risk = run(inputs, rules=PE.Rules(corr_risk_cap=(0.004, 0.003, 60)))
    assert (res.trades.reason == "trim_corr_risk").any()


# ---------------------------------------------------------------- regime engine
@pytest.fixture(scope="module")
def regime(raw):
    import regime_engine as RE
    data = {**raw, "BTCUSDT": daily(0, n=1500, start="2019-01-01")}
    data = {s: (daily(i, n=1500) if s != "BTCUSDT" else data[s]) for i, s in enumerate(COINS)}
    return RE, data, RE.frame(data, market=COINS)


def test_regime_features_are_causal(regime):
    RE, data, m = regime
    cut = m.index[900]
    trunc = {s: df[pd.to_datetime(df.timestamp, utc=True) <= cut] for s, df in data.items()}
    m2 = RE.frame(trunc, market=COINS)
    a = m.loc[:cut, RE.FEATURES + ["state_btc_up", "state_vol_tercile"]]
    b = m2.loc[:cut, RE.FEATURES + ["state_btc_up", "state_vol_tercile"]]
    pd.testing.assert_frame_equal(a, b)
    assert m2["y_stress"].iloc[-RE.H:].isna().all()             # labels need the future; none invented


def test_regime_walk_forward_purges_and_outputs_probabilities(regime, monkeypatch):
    RE, data, m = regime
    monkeypatch.setattr(RE, "TEST_YEARS", (2021, 2022))
    oos = RE.walk_forward(m)
    assert len(oos)
    for t, classes in RE.TARGETS.items():
        g = oos[oos.target == t]
        if not len(g):
            continue
        p = g[[f"p{j}_model" for j in range(len(classes))]].sum(axis=1)
        np.testing.assert_allclose(p, 1.0, atol=1e-9)
        assert (g.train_rows > 0).all()
        c = RE.confidence(g, "model", len(classes))
        assert ((c >= -1e-9) & (c <= 1 + 1e-9)).all()
    first_test = pd.Timestamp("2021-01-01", tz="UTC")
    assert (m.label_end[m.label_end < first_test - pd.Timedelta(days=RE.EMBARGO)].index < first_test).all()
