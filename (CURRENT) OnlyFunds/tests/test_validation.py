"""Validation engine tests (synthetic data only; never used for real-market conclusions)."""
import json

import numpy as np
import pandas as pd
import pytest

from core import validation as V
from core.backtester import BacktestConfig
from core.risk_manager import RiskLimits
from tests.synthetic import synthetic_regime


@pytest.fixture(scope="module")
def data():
    out = {}
    for k, kind in enumerate(("trend_up", "range", "random_walk")):
        df = synthetic_regime(kind, 3000, seed=k, interval="1h")  # starts 2025-01-01
        out[f"C{k}"] = df
    return out


BASE = BacktestConfig(interval="1h", risk=RiskLimits(kill_switch_file=""))
START, END = "2025-02-15", "2025-04-30"


def test_segment_keeps_history_and_window():
    df = synthetic_regime("range", 3000, seed=1, interval="1h")
    seg = V.segment(df, START, END, history=200)
    ts = pd.to_datetime(seg["timestamp"], utc=True)
    assert (ts < pd.Timestamp(START, tz="UTC")).sum() == 200
    assert ts.max() < pd.Timestamp(END, tz="UTC")


def test_run_period_trades_only_inside_window(data):
    res = V.run_period(data, BASE, "trend_following", None, START, END)
    for r in res.values():
        if len(r.trades):
            t = pd.to_datetime(r.trades["entry_time"], utc=True)
            assert t.min() >= pd.Timestamp(START, tz="UTC")
            assert pd.to_datetime(r.trades["exit_time"], utc=True).max() < pd.Timestamp(END, tz="UTC")


def test_pooled_metrics_and_bootstrap(data):
    res = V.run_period(data, BASE, "auto", None, START, END)
    p = V.pooled(res)
    if p["trades"]:
        all_r = pd.concat([r.trades["r_multiple"] for r in res.values() if len(r.trades)])
        assert p["expectancy_r"] == pytest.approx(all_r.mean())
        assert p["ci_low"] <= p["expectancy_r"] <= p["ci_high"]
    lo, hi = V.bootstrap_lower(np.array([1.0, -1.0] * 200))
    assert lo < 0 < hi
    lo2, _ = V.bootstrap_lower(np.full(300, 0.5) + np.random.default_rng(0).normal(0, 0.1, 300))
    assert lo2 > 0


def test_criteria_rules():
    good = {"trades": 300, "expectancy_r": 0.2, "ci_low": 0.05, "profit_factor": 1.3, "breadth": 0.7,
            "eligible_coins": 8, "exp_r_without_best_coin": 0.15}
    assert V.criteria(good) == (True, [])
    for k, v in (("trades", 50), ("expectancy_r", -0.1), ("ci_low", -0.01), ("profit_factor", 1.05),
                 ("breadth", 0.5), ("exp_r_without_best_coin", -0.02)):
        ok, why = V.criteria({**good, k: v})
        assert not ok and len(why) == 1
    assert V.walk_forward_pass(pd.DataFrame({"expectancy_r": [0.1, 0.2, -0.1, 0.05]}))[0]
    assert not V.walk_forward_pass(pd.DataFrame({"expectancy_r": [0.1, -0.2, -0.1, 0.05]}))[0]


def test_delisted_coin_skipped_after_its_data_ends(data):
    d = dict(data)
    d["DEAD"] = data["C0"].iloc[:800].reset_index(drop=True)   # ends ~2025-02-03
    res = V.run_period(d, BASE, "auto", None, START, END)
    assert "DEAD" not in res


def test_test_stage_guards(tmp_path, monkeypatch):
    from scripts import real_market_validation as R
    proto = tmp_path / "protocol.md"
    proto.write_text("rules v1")
    monkeypatch.setattr(R, "PROTOCOL", str(proto))
    monkeypatch.setattr(R, "OUT_DIR", str(tmp_path))
    monkeypatch.setattr(R, "FROZEN", str(tmp_path / "frozen.json"))
    args = type("A", (), {"rerun_test": False})()
    assert R.stage_test(args) == 2                                      # nothing frozen yet
    (tmp_path / "frozen.json").write_text(json.dumps({"protocol_sha256": V.protocol_fingerprint(str(proto)),
                                                      "symbols": [], "candidates": {}, "frozen_at_utc": "x"}))
    proto.write_text("rules v2 (edited after freezing)")
    assert R.stage_test(args) == 2                                      # protocol changed
    proto.write_text("rules v1")
    (tmp_path / "stage2_test.md").write_text("already run")
    assert R.stage_test(args) == 2                                      # one-shot rule
