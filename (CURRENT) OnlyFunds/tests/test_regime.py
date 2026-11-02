import numpy as np
import pandas as pd
import pytest

from core import regime as R
from tests.synthetic import synthetic_regime


@pytest.fixture(scope="module")
def df():
    return synthetic_regime("switching", 3000, seed=4, interval="1h")


def test_labels_valid_and_uncertain_allowed(df):
    d = R.detect(df)
    assert set(d["trend"].dropna()) <= set(R.TREND_CLASSES)
    assert set(d["volatility"].dropna()) <= {"low", "normal", "high"}
    assert set(d["vol_change"].dropna()) <= {"expanding", "contracting", "stable"}
    assert d["confidence"].dropna().between(0, 1).all()
    labels = d["label"].dropna()
    assert (labels == "uncertain").any() and (labels != "uncertain").any()
    low = d[(d["confidence"] < R.RegimeParams().min_confidence) & d["label"].notna()]
    assert (low["label"] == "uncertain").all()


def test_detector_is_causal(df):
    a = R.detect(df)
    t = df.copy()
    t.loc[2000:, ["Open", "High", "Low", "Close"]] *= 1.5
    b = R.detect(t)
    pd.testing.assert_frame_equal(a.iloc[:2000], b.iloc[:2000])


def test_persistence_reduces_transitions(df):
    e1 = R.evaluate(R.detect(df, R.RegimeParams(persistence=1)), df["Close"])
    e3 = R.evaluate(R.detect(df, R.RegimeParams(persistence=4)), df["Close"])
    assert e3["transitions_per_1000_bars"] < e1["transitions_per_1000_bars"]


def test_trend_direction_on_clear_regimes():
    up = synthetic_regime("trend_up", 2000, seed=1, interval="1h")
    down = synthetic_regime("trend_down", 2000, seed=1, interval="1h")
    tu, td = R.detect(up)["trend"].dropna(), R.detect(down)["trend"].dropna()
    assert tu.isin(["weak_up", "strong_up"]).mean() > tu.isin(["weak_down", "strong_down"]).mean()
    assert td.isin(["weak_down", "strong_down"]).mean() > td.isin(["weak_up", "strong_up"]).mean()


def test_expost_label_uses_future_and_evaluate_fields(df):
    lab = R.expost_trend(df["Close"], 48)
    assert lab.iloc[-48:].isna().all()                 # needs future bars: evaluation only
    ev = R.evaluate(R.detect(df), df["Close"])
    for k in ("transitions_per_1000_bars", "median_run_bars", "false_transition_share",
              "agreement_with_expost_direction", "chance_agreement", "median_detection_delay_bars", "uncertain_share"):
        assert k in ev and np.isfinite(ev[k])
