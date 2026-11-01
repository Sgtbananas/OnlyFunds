import pytest

from core.positions import ExitRules, check_exit, close_trade_record, open_position, apply_slippage
from core.risk_manager import RiskLimits, RiskManager

RULES = ExitRules(stop_atr_mult=1.0, tp_atr_mult=2.0, trail_atr_mult=1.0, use_trailing=True)


def pos():
    # entry 100, ATR 1 -> stop 99, tp 102, trailing 99
    return open_position("X", 1.0, 100.0, 0.001, 1.0, RULES, "t0")


def test_levels_from_fill():
    p = pos()
    assert (p.stop_loss, p.take_profit, p.trailing_stop) == (99.0, 102.0, 99.0)
    assert p.entry_fee == pytest.approx(0.1)


def test_stop_checked_before_take_profit_in_same_bar():
    assert check_exit(pos(), 100, 103, 98.5) == ("stop_loss", 99.0)


def test_gap_fills_at_open():
    assert check_exit(pos(), 98.0, 98.5, 97.0) == ("stop_loss", 98.0)
    assert check_exit(pos(), 103.0, 104.0, 102.5) == ("take_profit", 103.0)


def test_take_profit():
    assert check_exit(pos(), 100.5, 102.2, 100.1) == ("take_profit", 102.0)


def test_trailing_ratchets_only_after_bar_survives():
    p = pos()
    # Bar goes to 101.8 then down to 99.5: stop at bar start is 99 -> survives; trail moves to 100.8.
    assert check_exit(p, 100, 101.8, 99.5) == (None, None)
    assert p.trailing_stop == pytest.approx(100.8)
    assert p.effective_stop == pytest.approx(100.8)
    reason, price = check_exit(p, 101.0, 101.2, 100.5)
    assert reason == "trailing_stop" and price == pytest.approx(100.8)


def test_trailing_never_moves_down():
    p = pos()
    check_exit(p, 100, 101.5, 99.9)
    t = p.trailing_stop
    check_exit(p, 100.6, 100.9, 100.55)
    assert p.trailing_stop == t


def test_time_stop():
    p = pos()
    rules = ExitRules(1.0, 2.0, 1.0, False, max_hold_bars=2)
    p.trail_mult = 0
    assert check_exit(p, 100, 100.2, 99.8, rules) == (None, None)
    assert check_exit(p, 100, 100.2, 99.8, rules) == ("time_stop", None)


def test_trade_record_pnl_includes_both_fees():
    p = pos()
    rec = close_trade_record(p, 102.0, 0.001, "t1", "take_profit")
    assert rec["pnl"] == pytest.approx(102 - 0.102 - 100 - 0.1)
    assert rec["fees"] == pytest.approx(0.202)


def test_slippage_is_adverse():
    assert apply_slippage(100, "buy", 0.01) == pytest.approx(101)
    assert apply_slippage(100, "sell", 0.01) == pytest.approx(99)


# ------------------------------------------------------------------ risk
def rm(**kw):
    return RiskManager(RiskLimits(kill_switch_file="", **kw))


def test_risk_based_size():
    d = rm(max_position_pct=1.0, max_exposure_pct=1.0).position_size(1000, 1000, 100, 98)
    assert d.qty == pytest.approx(5.0)  # risk 10 USDT / 2 USDT stop distance
    assert "risk_per_trade" in d.reason


def test_size_capped_by_position_exposure_and_cash():
    r = rm(risk_per_trade=0.05, max_position_pct=0.2, max_exposure_pct=0.5)
    assert r.position_size(1000, 1000, 100, 99.9).qty == pytest.approx(2.0)  # 20% of equity
    assert r.position_size(1000, 1000, 100, 99.9, open_exposure=450).qty == pytest.approx(0.5)
    d = r.position_size(1000, 50, 100, 99.9, fee_rate=0.0)
    assert d.qty == pytest.approx(0.5) and "cash" in d.reason


def test_size_rejects_below_minimums():
    r = rm(min_notional=5.0)
    assert not r.position_size(100, 100, 100, 99, open_exposure=59.99).ok  # 0.01 USDT room left
    assert not r.position_size(1000, 1000, 100, 99, min_amount=100).ok
    assert not r.position_size(1000, 1000, 100, 101).ok  # stop above entry


def test_size_respects_precision():
    d = rm(max_position_pct=1, max_exposure_pct=1).position_size(1000, 1000, 3.0, 2.7, amount_precision=1)
    assert d.qty == pytest.approx(33.3)


def test_limits_validation():
    with pytest.raises(ValueError):
        RiskLimits(risk_per_trade=0.2).validate()
    with pytest.raises(ValueError):
        RiskLimits.from_mapping({"max_drawdown_pct": 0})


def test_can_open_rules(tmp_path):
    r = RiskManager(RiskLimits(max_open_positions=2, kill_switch_file=str(tmp_path / "KS")))
    assert r.can_open("A", [], 1000, 1000, 1000) == (True, "ok")
    assert r.can_open("A", ["A"], 1000, 1000, 1000)[0] is False
    assert r.can_open("C", ["A", "B"], 1000, 1000, 1000)[0] is False
    assert "daily" in r.can_open("C", [], 940, 1000, 1000)[1]
    assert "drawdown" in r.can_open("C", [], 840, 840, 1000)[1]
    r.engage_kill_switch("test")
    assert r.can_open("C", [], 1000, 1000, 1000) == (False, "kill switch active")
    r.clear_kill_switch()
    assert r.can_open("C", [], 1000, 1000, 1000)[0]


def test_legacy_config_mapping():
    lim = RiskLimits.from_mapping({"per_trade": 0.02})
    assert lim.risk_per_trade == 0.02
    assert RiskManager({"risk": {"per_trade": 0.003}}).limits.risk_per_trade == 0.003
