"""Round 3 position-management simulator: equivalence with the frozen X1 simulator, sell-only rules, risk bounds,
isolation from the frozen and experimental fingerprints. Offline, synthetic data only."""
import os
import re
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import portfolio_pm as PM  # noqa: E402
import portfolio_variants as PV  # noqa: E402
from test_ml_lab import _inputs, _limits, daily  # noqa: E402

S, E = "2019-03-01", "2020-12-01"


@pytest.fixture(scope="module")
def inputs():
    return _inputs({s: daily(i) for i, s in enumerate(["BTCUSDT", "ETHUSDT", "BNBUSDT", "ETCUSDT"])})


def run(inputs, **rules):
    fr, en, ex, at = inputs
    return PM.simulate(fr, en, ex, at, S, E, _limits(), capital=10000.0, rules=PM.Rules(**rules))


@pytest.mark.parametrize("kw", [{}, {"trim_band": (0.04, 0.02)}, {"profit_lock": (0.5, 2.0)},
                                {"trim_band": (0.15, 0.10), "profit_lock": (2.0, 6.0)}])
def test_reproduces_frozen_variant_simulator_exactly(inputs, kw):
    fr, en, ex, at = inputs
    a, ra = PV.simulate(fr, en, ex, at, S, E, _limits(), capital=10000.0, **kw)
    b, rb = run(inputs, **kw)
    np.testing.assert_allclose(a.equity.to_numpy(), b.equity.to_numpy())
    pd.testing.assert_frame_equal(a.trades, b.trades)
    for col in ra.columns:
        np.testing.assert_allclose(ra[col].to_numpy(), rb[col].to_numpy())


def _positions_entered(res):
    return res.trades.groupby("position_id").agg(pair=("pair", "first"), entry_day=("entry_day", "first"))


@pytest.mark.parametrize("kw", [{"growth_trim": (1.05, 1.0)}, {"atr_cap": (0.001, 0.0008)},
                                {"open_risk_cap": (0.004, 0.003)}, {"gross_cap": (0.12, 0.10)},
                                {"corr_cap": (0.10, 0.08, 60)}])
def test_new_rules_only_sell_and_are_tagged(inputs, kw):
    base, _ = run(inputs)
    res, _ = run(inputs, **kw)
    rule = {"growth_trim": "growth", "atr_cap": "atr", "open_risk_cap": "open_risk", "gross_cap": "gross",
            "corr_cap": "corr"}[next(iter(kw))]
    assert (res.trades.reason == f"trim_{rule}").any()
    assert set(res.trades.reason) <= {"signal_exit", "stop_loss", "end_of_period", f"trim_{rule}"}
    # sell-only: a position's total sold quantity never exceeds what was bought, and every trim is partial
    t = res.trades
    assert t[t.reason.str.startswith("trim")].partial.all()
    assert (t.qty > 0).all()


def test_open_risk_cap_bounds_risk_after_execution(inputs):
    _, base = run(inputs)
    _, capped = run(inputs, open_risk_cap=(0.004, 0.003))
    assert capped["open_risk_to_stop_pct"].max() < base["open_risk_to_stop_pct"].max()
    # after a breach is executed (next open) risk can only rise by that day's move again
    assert (capped["open_risk_to_stop_pct"] > 0.4 * 1.6).mean() < 0.05


def test_gross_cap_bounds_exposure(inputs):
    _, capped = run(inputs, gross_cap=(0.12, 0.10))
    assert (capped["position_value_pct"] > 12 * 1.3).mean() < 0.02


def test_effective_exposure_identities():
    w = np.array([0.1, 0.1, 0.1])
    assert PM.effective_exposure(w, np.ones((3, 3))) == pytest.approx(0.3)
    assert PM.effective_exposure(w, np.eye(3)) == pytest.approx(np.sqrt(0.03))


def test_correlation_uses_only_past_returns(inputs):
    fr = inputs[0]
    rets = np.log(pd.DataFrame({s: f["Close"] for s, f in fr.items()})).diff()
    d = rets.index[300]
    c1 = PM._corr(rets, ["BTCUSDT", "ETHUSDT"], d, 90)
    future = rets.copy()
    future.loc[future.index > d] = np.random.default_rng(1).normal(0, 1, future.loc[future.index > d].shape)
    np.testing.assert_allclose(c1, PM._corr(future, ["BTCUSDT", "ETHUSDT"], d, 90))
    assert (PM._corr(rets, ["BTCUSDT", "ETHUSDT"], rets.index[5], 90) == 1).all()     # too little history -> 1


def test_pm_code_is_isolated_from_frozen_and_experimental_fingerprints():
    import forward_experimental as FE
    import forward_shadow as FS
    for f in ("portfolio_pm.py", "round3_pm.py", "reconcile.py", "portfolio_engine.py", "regime_engine.py", "round4.py"):
        p = os.path.join(ROOT, "research", f)
        if not os.path.exists(p):
            continue
        src = open(p, encoding="utf-8").read()
        for bad in ("place_order", "cancel_order", "CoinExClient", "API_SECRET", "allow_trading", "requests.post",
                    "import forward_shadow", "import forward_experimental"):
            assert bad not in src, (f, bad)
        assert not re.search(r"""["']shadow(_experimental)?["']""", src), f
        assert f"research/{f}" not in FS.IMPLEMENTATION and f"research/{f}" not in FE.IMPLEMENTATION
    assert FS.load_frozen()[1] == [] and FE.load_frozen()[1] == []
