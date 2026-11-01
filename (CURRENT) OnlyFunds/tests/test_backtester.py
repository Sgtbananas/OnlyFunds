import numpy as np
import pandas as pd
import pytest

from core.backtester import BacktestConfig, run_backtest, run_out_of_sample, train_test_split_time
from core.core_signals import generate_signals
from core.positions import ExitRules
from core.risk_manager import RiskLimits
from tests.synthetic import synthetic_ohlcv


@pytest.fixture(scope="module")
def df():
    return synthetic_ohlcv(3000, seed=11)


def cfg(**kw):
    base = dict(strategy="auto", risk=RiskLimits(kill_switch_file="", max_drawdown_pct=1.0, max_daily_loss_pct=1.0,
                                                 max_cost_to_risk=10.0))  # mechanics test on 5m data: cost filter off
    base.update(kw)
    return BacktestConfig(**base)


def test_accounting_identity(df):
    res = run_backtest(df, cfg(close_at_end=True), "S")
    m = res.metrics
    assert m["trades"] > 10
    assert m["final_equity"] == pytest.approx(m["initial_capital"] + res.trades["pnl"].sum(), rel=1e-9)
    assert m["fees_paid"] == pytest.approx(res.trades["fees"].sum(), rel=1e-9)
    assert 0 <= m["win_rate_pct"] <= 100


def test_unrealised_pnl_reported_when_position_open(df):
    # Find a cut where a position is open, then check unrealised accounting.
    full = run_backtest(df, cfg(), "S")
    for k in range(2000, 3000, 7):
        res = run_backtest(df.iloc[:k], cfg(), "S")
        if res.open_position:
            m = res.metrics
            assert m["final_equity"] == pytest.approx(
                m["initial_capital"] + m["realised_pnl"] + m["unrealised_pnl"]
                + res.open_position["qty"] * df["Close"].iloc[k - 1] * cfg().fee_rate, rel=1e-9)
            return
    pytest.skip("no open position found at any cut")


def test_entries_execute_on_bar_after_signal(df):
    sig = generate_signals(df, strategy="auto")
    res = run_backtest(df, cfg(), "S", signals=sig)
    buys = res.decisions[res.decisions["action"] == "BUY"]
    assert len(buys)
    for _, b in buys.iterrows():
        assert sig["entry"].iloc[b["bar"] - 1], "buy without a signal on the previous bar"
        assert b["price"] == pytest.approx(df["Open"].iloc[b["bar"]] * (1 + cfg().slippage_pct))


def test_no_lookahead_future_changes_do_not_affect_past_trades(df):
    k = 2200
    a = run_backtest(df, cfg(), "S")
    tampered = df.copy()
    tampered.loc[k:, ["Open", "High", "Low", "Close"]] *= 1.3
    b = run_backtest(tampered, cfg(), "S")
    ta = a.trades[pd.to_datetime(a.trades["exit_time"]) < df["timestamp"].iloc[k - 1]]
    tb = b.trades[pd.to_datetime(b.trades["exit_time"]) < df["timestamp"].iloc[k - 1]]
    pd.testing.assert_frame_equal(ta.drop(columns="id").reset_index(drop=True),
                                  tb.drop(columns="id").reset_index(drop=True))


def test_reproducible(df):
    a = run_backtest(df, cfg(), "S").metrics
    b = run_backtest(df, cfg(), "S").metrics
    assert a == b


def test_fees_and_slippage_reduce_returns(df):
    free = run_backtest(df, cfg(fee_rate=0.0, slippage_pct=0.0), "S").metrics
    costly = run_backtest(df, cfg(fee_rate=0.004, slippage_pct=0.002), "S").metrics
    assert costly["total_return_pct"] < free["total_return_pct"]


def test_single_trade_exact_pnl():
    """Hand-built path: entry signal forced, price rises to take-profit."""
    n = 120
    close = np.full(n, 100.0)
    data = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC"),
                         "Open": close, "High": close + 0.5, "Low": close - 0.5, "Close": close})
    data.loc[101:, ["Open", "Close"]] = 100.0
    data.loc[103, "High"] = 200.0  # hits any TP
    sig = generate_signals(data)
    sig["entry"] = False
    sig["exit"] = False
    sig.loc[100, "entry"] = True
    atr = sig["ATR"].iloc[100]
    c = cfg(fee_rate=0.001, slippage_pct=0.0, exit_rules=ExitRules(1.0, 2.0, 1.0, use_trailing=False),
            risk=RiskLimits(kill_switch_file="", risk_per_trade=0.01, max_position_pct=1.0, max_exposure_pct=1.0,
                            max_cost_to_risk=10.0))
    res = run_backtest(data, c, "S", signals=sig)
    assert len(res.trades) == 1
    t = res.trades.iloc[0]
    stop = 100 - 1.0 * atr
    loss_per_unit = 100 * 1.001 - stop * (1 - c.stop_slippage_pct) * 0.999  # fees + stop slippage budgeted
    qty = min(1000 * 0.01 / loss_per_unit, 1000 / (100 * 1.001))
    tp = 100 + 2 * atr
    assert t["exit_reason"] == "take_profit"
    assert t["qty"] == pytest.approx(qty)
    assert t["exit_price"] == pytest.approx(tp)
    assert t["pnl"] == pytest.approx(qty * tp * 0.999 - qty * 100 * 1.001)


def test_ml_gate_blocks_entries(df):
    zeros = pd.Series(0.0, index=df.index)
    res = run_backtest(df, cfg(), "S", ml_proba=zeros)
    assert res.metrics["trades"] == 0
    assert (res.decisions["action"] == "FILTERED").any()


def test_drawdown_halt_stops_new_entries(df):
    res = run_backtest(df, cfg(risk=RiskLimits(kill_switch_file="", max_drawdown_pct=0.005, max_cost_to_risk=10.0)), "S")
    if res.metrics["halted"]:
        halt_bar = res.decisions.loc[res.decisions["action"] == "HALT", "bar"].iloc[0]
        assert not ((res.decisions["action"] == "BUY") & (res.decisions["bar"] > halt_bar + 1)).any()


def test_out_of_sample_split(df):
    r = run_out_of_sample(df, cfg(), "S", train_frac=0.7)
    assert set(r) == {"in_sample", "out_of_sample"}
    oos_first_trade = r["out_of_sample"].trades["entry_time"].min()
    cut_time = df["timestamp"].iloc[int(len(df) * 0.7)]
    assert pd.Timestamp(oos_first_trade) >= cut_time
    tr, te = train_test_split_time(df, 0.7, gap=10)
    assert len(tr) + len(te) + 10 == len(df)


def test_rejects_bad_data():
    with pytest.raises(ValueError):
        run_backtest(pd.DataFrame(), cfg())
