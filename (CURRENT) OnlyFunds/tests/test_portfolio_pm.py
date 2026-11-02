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


def missing_bar_case():
    """Two held coins; A jumps +40% at the close of day 40 and then has no bar on day 41. Its risk must still count
    towards the open-risk cap on day 41 (it cannot be sold then, so B is cut instead)."""
    from core.risk_manager import RiskLimits
    idx = pd.date_range("2020-01-01", periods=60, freq="D", tz="UTC")
    def frame(close):
        c = pd.Series(close, index=idx, dtype=float)
        o = c.shift(1).fillna(c.iloc[0])
        return pd.DataFrame({"Open": o, "High": np.maximum(o, c) * 1.001, "Low": np.minimum(o, c) * 0.999,
                             "Close": c, "Volume": 1e9})
    wiggle = 0.05 * np.sin(np.arange(60))                      # avoids zero-variance correlations
    a = np.r_[np.full(40, 100.0), np.full(20, 140.0)] + wiggle
    fr = {"AUSDT": frame(a), "BUSDT": frame(100.0 + np.cos(np.arange(60)) * 0.05)}
    jump, gap = idx[40], idx[41]
    fr["AUSDT"] = fr["AUSDT"].drop(index=gap)
    en = {p: pd.Series(f.index == idx[20], index=f.index) for p, f in fr.items()}
    ex = {p: pd.Series(False, index=f.index) for p, f in fr.items()}
    at = {p: pd.Series(2.0, index=f.index) for p, f in fr.items()}
    lim = RiskLimits(risk_per_trade=0.005, max_position_pct=0.10, max_exposure_pct=0.30, max_open_positions=3,
                     max_daily_loss_pct=1.0, max_drawdown_pct=1.0, min_notional=1.0, max_cost_to_risk=0.25,
                     kill_switch_file="")
    return (fr, en, ex, at, str(idx[0].date()), str((idx[-1] + pd.Timedelta(days=1)).date()), lim), jump, gap


def test_open_risk_cap_counts_a_held_coin_with_no_bar():
    (fr, en, ex, at, s, e, lim), jump, gap = missing_bar_case()
    res, risk = PM.simulate(fr, en, ex, at, s, e, lim, capital=10000.0, rules=PM.Rules(open_risk_cap=(0.03, 0.025)))
    assert risk.loc[jump, "open_risk_to_stop_pct"] > 3.0 and risk.loc[gap, "open_risk_to_stop_pct"] > 3.0
    t = res.trades
    b_cut = t[(t.pair == "BUSDT") & (t.reason == "trim_open_risk")]
    assert len(b_cut) and (b_cut.exit_day == gap + pd.Timedelta(days=1)).all()     # B cut at the next open
    a_cut = t[(t.pair == "AUSDT") & (t.reason == "trim_open_risk")]
    assert len(a_cut) and a_cut.exit_day.min() > gap                                # A only once it trades again


def stale_case():
    """missing_bar_case plus coin C, which signals at the close of day 40 (fill due at the open of day 41, when A is
    unpriced), at the close of day 41 (A unpriced) and at the close of day 45 (everything priced)."""
    (fr, en, ex, at, s, e, lim), jump, gap = missing_bar_case()
    idx = fr["BUSDT"].index
    fr["CUSDT"] = fr["BUSDT"].copy()
    en["CUSDT"] = pd.Series(idx.isin([idx[40], idx[41], idx[45]]), index=idx)
    ex["CUSDT"] = pd.Series(False, index=idx)
    at["CUSDT"] = pd.Series(2.0, index=idx)
    return (fr, en, ex, at, s, e, lim), idx


@pytest.mark.parametrize("sim", ["pm", "engine"])
def test_unpriced_position_flags_risk_and_blocks_new_exposure(sim):
    import portfolio_engine as PE
    (fr, en, ex, at, s, e, lim), idx = stale_case()
    mod = PM if sim == "pm" else PE
    res, risk = mod.simulate(fr, en, ex, at, s, e, lim, capital=10000.0)
    gap = idx[41]
    assert not risk.loc[gap, "risk_reliable"] and risk.loc[gap, "unpriced"] == "AUSDT"
    assert risk.loc[gap, "unpriced_positions"] == 1 and risk.loc[gap, "max_price_age_days"] == 1
    assert risk.drop(index=gap)["risk_reliable"].all() and risk.drop(index=gap)["max_price_age_days"].max() == 0
    assert risk.loc[gap, "entries_blocked_unreliable"] == 2       # the fill at the open and the decision at the close
    c = res.trades[res.trades.pair == "CUSDT"]
    assert len(c) and c.entry_day.min() == idx[46]                  # C enters only once every price is current
    assert risk.loc[gap, "open_risk_to_stop_pct"] > 0               # A is still in aggregate risk
    assert risk.loc[gap, "n_open"] == 2


def test_gross_cap_counts_an_unpriced_position():
    (fr, en, ex, at, s, e, lim), jump, gap = missing_bar_case()
    res, risk = PM.simulate(fr, en, ex, at, s, e, lim, capital=10000.0, rules=PM.Rules(gross_cap=(0.105, 0.10)))
    t = res.trades
    # gap day: A (about 7% of equity) is unpriced and unsellable, B about 4%; together above 10.5%, so B is cut at
    # the next open. Counting only priced positions (B alone) would not have cut anything.
    assert risk.loc[gap, "position_value_pct"] > 10.5
    b = t[(t.pair == "BUSDT") & (t.exit_day == gap + pd.Timedelta(days=1))]
    assert len(b) and (b.reason == "trim_gross").all()
    assert not ((t.pair == "AUSDT") & (t.exit_day == gap)).any()
