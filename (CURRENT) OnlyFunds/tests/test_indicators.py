import numpy as np
import pandas as pd
import pytest

from core import indicators as ind
from tests.synthetic import synthetic_ohlcv


@pytest.fixture
def df():
    return synthetic_ohlcv(600, seed=7)


def ref_ema(x, span):
    a = 2 / (span + 1)
    out = np.empty(len(x))
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    out[: span - 1] = np.nan
    return out


def ref_wilder(x, n):
    out = np.empty(len(x))
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = out[i - 1] + (x[i] - out[i - 1]) / n
    out[: n - 1] = np.nan
    return out


def test_ema_matches_recursive_definition(df):
    c = df["Close"].to_numpy()
    np.testing.assert_allclose(ind.ema(df["Close"], 21).to_numpy(), ref_ema(c, 21), rtol=1e-12, equal_nan=True)


def test_rsi_matches_wilder_reference(df):
    c = df["Close"].to_numpy()
    d = np.diff(c, prepend=np.nan)
    gain, loss = np.clip(np.nan_to_num(d), 0, None), np.clip(-np.nan_to_num(d), 0, None)
    gain[0] = loss[0] = np.nan
    # Wilder smoothing starting from the first real delta (index 1)
    g = ref_wilder(gain[1:], 14)
    l = ref_wilder(loss[1:], 14)
    expected = np.r_[np.nan, 100 - 100 / (1 + g / l)]
    got = ind.rsi(df["Close"], 14).to_numpy()
    np.testing.assert_allclose(got[20:], expected[20:], rtol=1e-10)
    assert np.nanmin(got) >= 0 and np.nanmax(got) <= 100


def test_rsi_edge_cases():
    up = pd.Series(np.arange(1, 60, dtype=float))
    assert ind.rsi(up, 14).iloc[-1] == 100
    flat = pd.Series(np.full(60, 5.0))
    assert ind.rsi(flat, 14).iloc[-1] == 50


def test_macd_definition(df):
    line, sig, hist = ind.macd(df["Close"], 12, 26, 9)
    c = df["Close"].to_numpy()
    expected_line = ref_ema(c, 12) - ref_ema(c, 26)
    np.testing.assert_allclose(line.to_numpy()[30:], expected_line[30:], rtol=1e-10)
    np.testing.assert_allclose(hist.to_numpy(), (line - sig).to_numpy(), equal_nan=True)
    with pytest.raises(ValueError):
        ind.macd(df["Close"], 26, 12, 9)


def test_bollinger_definition(df):
    mid, up, lo, width, pctb = ind.bollinger(df["Close"], 20, 2.0)
    c = df["Close"].to_numpy()
    i = 100
    w = c[i - 19:i + 1]
    assert mid.iloc[i] == pytest.approx(w.mean())
    assert up.iloc[i] == pytest.approx(w.mean() + 2 * w.std(ddof=0))
    assert lo.iloc[i] == pytest.approx(w.mean() - 2 * w.std(ddof=0))
    assert pctb.iloc[i] == pytest.approx((c[i] - lo.iloc[i]) / (up.iloc[i] - lo.iloc[i]))


def test_atr_matches_wilder_true_range(df):
    h, l, c = (df[k].to_numpy() for k in ("High", "Low", "Close"))
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    tr[0] = h[0] - l[0]
    np.testing.assert_allclose(ind.atr(df["High"], df["Low"], df["Close"], 14).to_numpy(),
                               ref_wilder(tr, 14), rtol=1e-10, equal_nan=True)


def test_trend_score_bounded_and_directional():
    up = synthetic_ohlcv(400, seed=1, drift=0.002, cycle=0)
    down = synthetic_ohlcv(400, seed=1, drift=-0.002, cycle=0)
    tu = ind.add_indicators(up)["trend_score"].iloc[-50:]
    td = ind.add_indicators(down)["trend_score"].iloc[-50:]
    assert tu.between(-1, 1).all() and td.between(-1, 1).all()
    assert tu.mean() > 0.3 and td.mean() < -0.3


def test_indicators_are_causal(df):
    """Changing future bars must not change any past indicator value."""
    full = ind.add_indicators(df)
    k = 400
    tampered = df.copy()
    tampered.loc[k:, ["Open", "High", "Low", "Close"]] *= 1.5
    t = ind.add_indicators(tampered)
    cols = [c for c in full.columns if c not in df.columns]
    pd.testing.assert_frame_equal(full.loc[: k - 1, cols], t.loc[: k - 1, cols])


def test_add_indicators_requires_ohlc():
    with pytest.raises(ValueError):
        ind.add_indicators(pd.DataFrame({"Close": [1.0, 2.0]}))
    assert ind.add_indicators(pd.DataFrame()).empty


def test_validate_ohlcv():
    assert not ind.validate_ohlcv(None)
    bad = synthetic_ohlcv(10)
    bad.loc[3, "Close"] = 0
    assert not ind.validate_ohlcv(bad)
    assert ind.validate_ohlcv(synthetic_ohlcv(10))
