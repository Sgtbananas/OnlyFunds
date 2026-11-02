"""Second-pass audit: hand-built price paths that pin down every execution assumption.

Setup used throughout: 120 flat bars at 100 with High/Low = 100 +/- 0.5, so ATR = 1.0
exactly. An entry signal is forced on bar 100, so the entry fills at the open of bar 101.
Exit rules: stop 1 ATR, take-profit 2 ATR, trailing 1 ATR unless stated.
"""
import numpy as np
import pandas as pd
import pytest

from core.backtester import BacktestConfig, run_backtest, run_out_of_sample
from core.core_signals import generate_signals
from core.positions import ExitRules
from core.risk_manager import RiskLimits
from tests.synthetic import synthetic_ohlcv

N = 140
SIG_BAR, ENTRY_BAR = 100, 101


def flat(n=N, price=100.0):
    ts = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({"timestamp": ts, "created_at": ts.asi8 // 10**6, "Open": price, "High": price + 0.5,
                         "Low": price - 0.5, "Close": price, "Volume": 1e9})


def bar(df, i, o, h, l, c):
    df.loc[i, ["Open", "High", "Low", "Close"]] = [o, h, l, c]


def forced(df, entry_bars=(SIG_BAR,)):
    sig = generate_signals(df)
    sig["entry"] = False
    sig["exit"] = False
    for b in entry_bars:
        sig.loc[b, "entry"] = True
    sig["ATR"] = 1.0  # pin ATR so the scripted bars don't change levels
    return sig


def cfg(fee=0.0, slip=0.0, stop_slip=None, trailing=False, **risk):
    r = dict(kill_switch_file="", risk_per_trade=0.01, max_position_pct=1.0, max_exposure_pct=1.0,
             max_daily_loss_pct=1.0, max_drawdown_pct=1.0, max_cost_to_risk=10.0)  # mechanics test on 5m data: cost filter off
    r.update(risk)
    kw = {"stop_slippage_pct": 0.0 if stop_slip is None else stop_slip}
    return BacktestConfig(fee_rate=fee, slippage_pct=slip, risk=RiskLimits(**r),
                          exit_rules=ExitRules(1.0, 2.0, 1.0, use_trailing=trailing), **kw)


def one_trade(df, c):
    res = run_backtest(df, c, "T", signals=forced(df))
    assert len(res.trades) == 1, res.decisions
    return res, res.trades.iloc[0]


# ------------------------------------------------------------- timing
def test_entry_fills_at_next_open_not_signal_close():
    df = flat()
    df.loc[SIG_BAR, "Close"] = 100.4
    bar(df, ENTRY_BAR, 100.2, 100.6, 99.8, 100.3)
    bar(df, 105, 100, 103, 99.9, 102.5)
    res, t = one_trade(df, cfg())
    assert t["entry_price"] == pytest.approx(100.2)
    assert pd.Timestamp(t["entry_time"]) == df["timestamp"][ENTRY_BAR]


def test_signal_exit_fills_at_next_open():
    df = flat()
    sig = forced(df)
    sig.loc[110, "exit"] = True
    bar(df, 111, 100.3, 100.5, 99.9, 100.1)
    res = run_backtest(df, cfg(), "T", signals=sig)
    t = res.trades.iloc[0]
    assert t["exit_reason"] == "signal_exit" and t["exit_price"] == pytest.approx(100.3)
    assert pd.Timestamp(t["exit_time"]) == df["timestamp"][111]


# ------------------------------------------------------------- gaps / same candle
def test_gap_down_through_stop_fills_at_open_not_stop():
    df = flat()
    bar(df, 103, 97.0, 97.5, 96.0, 97.2)   # stop is 99; opens at 97
    _, t = one_trade(df, cfg())
    assert t["exit_reason"] == "stop_loss" and t["exit_price"] == pytest.approx(97.0)


def test_gap_up_through_take_profit_fills_at_open():
    df = flat()
    bar(df, 103, 103.0, 103.5, 102.8, 103.2)   # tp is 102
    _, t = one_trade(df, cfg())
    assert t["exit_reason"] == "take_profit" and t["exit_price"] == pytest.approx(103.0)


def test_stop_and_target_in_same_candle_assumes_stop():
    df = flat()
    bar(df, 103, 100, 102.5, 98.5, 101)
    _, t = one_trade(df, cfg())
    assert t["exit_reason"] == "stop_loss" and t["exit_price"] == pytest.approx(99.0)


def test_entry_bar_itself_can_stop_out():
    df = flat()
    bar(df, ENTRY_BAR, 100, 100.2, 98.0, 98.5)
    _, t = one_trade(df, cfg())
    assert t["exit_reason"] == "stop_loss" and pd.Timestamp(t["exit_time"]) == df["timestamp"][ENTRY_BAR]


# ------------------------------------------------------------- trailing
def test_trailing_stop_sequence():
    df = flat()
    c = cfg(trailing=True)
    c.exit_rules.tp_atr_mult = 50.0
    bar(df, 102, 100, 101.5, 99.6, 101.4)   # trail -> 100.5 after bar
    bar(df, 103, 101.4, 102.8, 101.0, 102.6)  # trail -> 101.8
    bar(df, 104, 102.6, 102.7, 101.5, 101.6)  # low 101.5 <= 101.8 -> exit at 101.8
    _, t = one_trade(df, c)
    assert t["exit_reason"] == "trailing_stop" and t["exit_price"] == pytest.approx(101.8)


# ------------------------------------------------------------- costs
def test_fees_and_slippage_on_both_legs():
    df = flat()
    bar(df, 104, 100, 102.5, 99.9, 102.2)
    _, t = one_trade(df, cfg(fee=0.002, slip=0.001))
    entry = 100 * 1.001
    exit_ = (entry + 2.0) * 0.999          # tp level is set from the actual fill
    assert t["entry_price"] == pytest.approx(entry)
    assert t["exit_price"] == pytest.approx(exit_)
    assert t["fees"] == pytest.approx(t["qty"] * (entry + exit_) * 0.002)
    assert t["pnl"] == pytest.approx(t["qty"] * (exit_ * 0.998 - entry * 1.002))


def test_stop_exit_uses_wider_stop_slippage():
    df = flat()
    bar(df, 103, 100, 100.2, 98.0, 98.4)
    _, t = one_trade(df, cfg(slip=0.0005, stop_slip=0.003))
    assert t["exit_price"] == pytest.approx((100 * 1.0005 - 1.0) * (1 - 0.003))  # stop set from the slipped fill


def test_realised_stop_loss_never_exceeds_risk_budget():
    """Sizing must budget for both fees and stop slippage, not just the stop distance."""
    df = flat()
    bar(df, 103, 100, 100.2, 98.0, 98.4)
    res, t = one_trade(df, cfg(fee=0.002, slip=0.0005, stop_slip=0.003))
    budget = 1000 * 0.01
    assert -t["pnl"] <= budget + 1e-9
    assert -t["pnl"] > budget * 0.95, "sizing should use most of the budget, not be needlessly small"


# ------------------------------------------------------------- liquidity
def test_position_capped_by_volume_participation():
    df = flat()
    df["Volume"] = 2.0                      # 2 units per bar -> 10% participation = 0.2 units
    bar(df, 104, 100, 102.5, 99.9, 102.2)
    c = cfg()
    c.max_volume_participation = 0.10
    _, t = one_trade(df, c)
    assert t["qty"] == pytest.approx(0.2)


# ------------------------------------------------------------- accounting / metrics
def test_drawdown_counts_intrabar_lows():
    df = flat()
    c = cfg(risk_per_trade=0.05)
    c.exit_rules.stop_atr_mult = 30.0
    c.exit_rules.tp_atr_mult = 30.0
    bar(df, 103, 100, 100.2, 80.0, 100.0)   # deep wick, closes flat
    res = run_backtest(df, c, "T", signals=forced(df))
    qty = res.open_position["qty"]
    expected_dd = qty * 20.0 / 1000 * 100
    assert res.metrics["max_drawdown_pct"] == pytest.approx(expected_dd, rel=1e-6)


def test_buy_and_hold_aligned_to_trading_window_and_net_of_fees():
    df = synthetic_ohlcv(3000, seed=9)
    c = BacktestConfig(fee_rate=0.002, slippage_pct=0.001)
    oos = run_out_of_sample(df, c, "S")["out_of_sample"]
    cut = int(len(df) * 0.7)
    o, c_ = df["Open"].iloc[cut + 1], df["Close"].iloc[-1]
    expected = (c_ * 0.999 * 0.998 / (o * 1.001 * 1.002) - 1) * 100
    assert oos.metrics["buy_and_hold_return_pct"] == pytest.approx(expected, rel=1e-6)


def test_expectancy_and_frequency_metrics():
    res = run_backtest(synthetic_ohlcv(3000, seed=2), BacktestConfig(risk=RiskLimits(kill_switch_file="",
                                                                                   max_cost_to_risk=10.0)), "S")
    m = res.metrics
    if m["trades"]:
        assert m["expectancy"] == pytest.approx(res.trades["pnl"].mean())
        assert m["trades_per_day"] > 0
        assert sum(v["trades"] for v in m["by_regime"].values()) == m["trades"]


def test_missing_candles_are_reported():
    df = synthetic_ohlcv(600, seed=1).drop(index=range(300, 305)).reset_index(drop=True)
    m = run_backtest(df, BacktestConfig(), "S").metrics
    assert m["missing_candles"] == 5


def test_daily_loss_limit_blocks_rest_of_day():
    df = flat(N + 30)
    bar(df, 103, 90.0, 90.2, 89.0, 90.0)    # gap down through the stop: ~10% loss on the position
    sig = forced(df, entry_bars=(SIG_BAR, 120))
    c = cfg(risk_per_trade=0.05, max_daily_loss_pct=0.03)
    res = run_backtest(df, c, "T", signals=sig)
    assert len(res.trades) == 1
    assert (res.decisions["action"] == "BLOCKED").any()
    assert "daily loss" in res.decisions.loc[res.decisions["action"] == "BLOCKED", "reason"].iloc[0]


def test_cash_never_negative_and_equity_consistent():
    res = run_backtest(synthetic_ohlcv(3000, seed=5), BacktestConfig(close_at_end=True, risk=RiskLimits(
        kill_switch_file="", max_cost_to_risk=10.0)), "S")
    assert (res.equity["cash"] >= -1e-9).all()
    assert res.metrics["final_equity"] == pytest.approx(1000 + res.trades["pnl"].sum())
