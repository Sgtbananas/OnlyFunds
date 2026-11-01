"""Second-pass audit of the ML filter: costs, expectancy and leakage."""
import numpy as np
import pandas as pd
import pytest

from core import ml_filter
from core.core_signals import generate_signals
from core.positions import ExitRules
from tests.synthetic import synthetic_ohlcv

RULES = ExitRules(1.0, 2.0, 1.0)


def flat(n=12):
    return pd.DataFrame({"Open": [100.0] * n, "High": [100.5] * n, "Low": [99.5] * n, "Close": [100.0] * n})


def test_barrier_outcome_r_multiples_after_costs():
    df = flat()
    df.loc[3, "High"] = 103            # TP for t=0..2
    df.loc[6, "Low"] = 97              # stop for t=3..5
    atr = pd.Series(1.0, index=df.index)
    r = ml_filter.barrier_outcomes(df, atr, RULES, horizon=3, fee_rate=0.001, stop_slippage_pct=0.002)
    tp_r = (102 * 0.999 - 100 * 1.001) / 1.0
    stop_r = (99 * 0.998 * 0.999 - 100 * 1.001) / 1.0
    assert r[0] == pytest.approx(tp_r) and r[3] == pytest.approx(stop_r)
    # timeout: exit at the close of the last horizon bar
    assert r[6] == pytest.approx((100 * 0.999 - 100 * 1.001) / 1.0)
    assert np.isnan(r[9])


def test_outcome_and_label_agree_on_direction():
    df = synthetic_ohlcv(1500, seed=12)
    sig = generate_signals(df)
    y = ml_filter.triple_barrier_labels(sig, sig["ATR"], RULES, 24)
    r = ml_filter.barrier_outcomes(sig, sig["ATR"], RULES, 24, 0.0, 0.0)
    ok = y.notna() & r.notna()
    assert (r[ok & (y == 1)] > 0).all()           # every TP label has a positive outcome


def test_edge_check_rejects_higher_win_rate_with_lower_expectancy():
    assert not ml_filter.edge_check(0.70, 50, expectancy_r_accepted=0.05, expectancy_r_all=0.10)
    assert not ml_filter.edge_check(0.70, 50, expectancy_r_accepted=-0.01, expectancy_r_all=-0.20)
    assert not ml_filter.edge_check(0.52, 50, 0.3, 0.1)
    assert not ml_filter.edge_check(0.70, 5, 0.3, 0.1)
    assert not ml_filter.edge_check(float("nan"), 50, 0.3, 0.1)
    assert ml_filter.edge_check(0.60, 50, 0.20, 0.05)


def test_features_are_causal():
    df = synthetic_ohlcv(1200, seed=3)
    a = ml_filter.build_features(generate_signals(df))
    t = df.copy()
    t.loc[900:, ["Open", "High", "Low", "Close"]] *= 1.4
    b = ml_filter.build_features(generate_signals(t))
    pd.testing.assert_frame_equal(a.iloc[:900], b.iloc[:900])


def test_report_contains_cost_aware_metrics(tmp_path):
    frames = {"A": synthetic_ohlcv(3000, seed=21), "B": synthetic_ohlcv(3000, seed=22)}
    rep = ml_filter.train_filter(frames, RULES, save_path=None)["report"]
    for k in ("validation_expectancy_r_all", "validation_expectancy_r_accepted", "validation_accepted_trades"):
        assert k in rep
    assert rep["useful"] == ml_filter.edge_check(rep["validation_auc"], rep["validation_accepted_trades"],
                                                 rep["validation_expectancy_r_accepted"],
                                                 rep["validation_expectancy_r_all"])


def test_old_model_versions_are_ignored(tmp_path):
    import joblib
    path = tmp_path / "m.pkl"
    joblib.dump({"model": None, "features": [], "report": {"version": 2}}, path)
    assert ml_filter.load_bundle(str(path)) is None
