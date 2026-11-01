import numpy as np
import pandas as pd
import pytest

from core.core_signals import STRATEGY_NAMES, REGIMES, generate_signals, generate_signal, smooth_signal
from core.indicators import warmup_bars
from tests.synthetic import synthetic_ohlcv


@pytest.fixture(scope="module")
def df():
    return synthetic_ohlcv(1500, seed=3)


@pytest.mark.parametrize("strategy", STRATEGY_NAMES)
def test_signals_shape_and_bounds(df, strategy):
    s = generate_signals(df, strategy=strategy)
    assert len(s) == len(df)
    assert s["signal"].between(-1, 1).all()
    assert s["confidence"].between(0, 1).all()
    assert not (s["entry"] & s["exit"]).any(), "contradictory entry and exit on the same bar"
    assert not s["entry"].iloc[: warmup_bars() - 30].any(), "entries during indicator warm-up"
    assert set(s["regime"].dropna()) <= set(REGIMES)


@pytest.mark.parametrize("strategy", STRATEGY_NAMES)
def test_signals_are_causal(df, strategy):
    """Signals for bars < k must not depend on bars >= k."""
    k = 1000
    full = generate_signals(df, strategy=strategy)
    tampered = df.copy()
    tampered.loc[k:, ["Open", "High", "Low", "Close"]] *= 0.7
    t = generate_signals(tampered, strategy=strategy)
    cols = ["signal", "confidence", "regime", "strategy", "entry", "exit", "confirmations"]
    pd.testing.assert_frame_equal(full.loc[: k - 1, cols], t.loc[: k - 1, cols])


def test_auto_never_enters_in_downtrend():
    down = synthetic_ohlcv(800, seed=5, drift=-0.003, cycle=0)
    s = generate_signals(down, strategy="auto")
    assert not s.loc[s["regime"] == "trending_down", "entry"].any()


def test_entry_requires_confirmations(df):
    s = generate_signals(df, strategy="trend_following", params={"min_confirmations": 3})
    assert (s.loc[s["entry"], "confirmations"] >= 3).all()


def test_unknown_strategy_rejected(df):
    with pytest.raises(ValueError):
        generate_signals(df, strategy="moon")


def test_legacy_helpers(df):
    sig = generate_signal(df)
    assert isinstance(sig, pd.Series) and len(sig) == len(df)
    sm = smooth_signal(sig, 3)
    assert sm.iloc[10] == pytest.approx(sig.iloc[8:11].mean())
