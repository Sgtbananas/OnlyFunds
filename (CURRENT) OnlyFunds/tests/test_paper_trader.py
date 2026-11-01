"""Paper trading tests: simulated clock, synthetic candles, network blocked."""
import os
from datetime import timedelta

import pandas as pd
import pytest
import requests

from core.paper_trader import LedgerBusy, PaperSettings, PaperTrader, ledger_lock
from core.positions import ExitRules
from core.risk_manager import RiskLimits
from core.trade_execution import (LIVE_TRADING_LOCKED, LiveTradingDisabledError, PaperExecutor,
                                  get_executor, place_live_order)
from tests.synthetic import synthetic_ohlcv


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network access attempted during paper trading test")
    monkeypatch.setattr(requests.Session, "request", boom)


class Market:
    """Replays synthetic candles as if time were passing."""

    def __init__(self, n=2500, seed=21, start_bar=400):
        self.df = synthetic_ohlcv(n, seed=seed)
        self.i = start_bar

    def now(self):
        # Just after bar i closed
        return (self.df["timestamp"].iloc[self.i] + timedelta(minutes=5, seconds=1)).to_pydatetime()

    def klines(self, pair, interval, limit):
        return self.df.iloc[max(0, self.i + 1 - limit): self.i + 1].reset_index(drop=True)

    def price(self, pair):
        return float(self.df["Open"].iloc[self.i + 1])  # current price = next bar open


def make(tmp_path, market, executor=None, **kw):
    s = PaperSettings(interval="5m", lookback=400, risk=RiskLimits(kill_switch_file=str(tmp_path / "KS"),
                                                                    max_drawdown_pct=0.5, max_daily_loss_pct=0.5,
                                                                    max_cost_to_risk=10.0), # mechanics test on 5m data: cost filter off
                      **kw)
    return PaperTrader(s, str(tmp_path), data_fn=market.klines, price_fn=market.price,
                       executor=executor, now_fn=market.now)


def run(trader, market, steps, pairs=("SYN",)):
    out = []
    for _ in range(steps):
        out += trader.step(list(pairs))
        market.i += 1
    return out


def open_any_position(trader, market, max_steps=2000):
    for _ in range(max_steps):
        trader.step(["SYN"])
        market.i += 1
        if trader.ledger["positions"]:
            return
    raise AssertionError("no position opened")


def test_paper_session_trades_and_reconciles(tmp_path):
    m = Market()
    t = make(tmp_path, m)
    decisions = run(t, m, 1000)
    actions = pd.Series([d["action"] for d in decisions])
    assert (actions == "BUY").sum() > 3 and (actions == "SELL").sum() > 3
    assert t.reconcile() == []
    met = t.metrics()
    assert met["trades"] == len(t.ledger["trades"]) > 0
    assert met["fees_paid"] > 0
    assert met["equity"] == pytest.approx(t.equity())


def test_one_decision_per_bar(tmp_path):
    m = Market()
    t = make(tmp_path, m)
    first = t.step(["SYN"])
    again = t.step(["SYN"])  # same bar, nothing new
    assert len(first) == 1 and first[0]["action"] in ("WAIT", "BUY", "SKIP", "BLOCKED")
    assert all(d["action"] not in ("BUY", "WAIT") for d in again)


def test_no_duplicate_positions(tmp_path):
    m = Market()
    t = make(tmp_path, m)
    run(t, m, 400, pairs=("SYN", "SYN"))
    assert t.reconcile() == []
    assert len(t.ledger["positions"]) <= 1


def test_ledger_persists_across_restart(tmp_path):
    m = Market()
    t = make(tmp_path, m)
    run(t, m, 600)
    snapshot = (t.ledger["cash"], len(t.ledger["trades"]), dict(t.ledger["positions"]))
    t2 = make(tmp_path, m)
    assert (t2.ledger["cash"], len(t2.ledger["trades"]), dict(t2.ledger["positions"])) == snapshot
    run(t2, m, 300)
    assert t2.reconcile() == []


def test_corrupt_ledger_can_be_quarantined_explicitly(tmp_path):
    from core.paper_trader import LedgerCorrupt, quarantine_ledger
    (tmp_path / "paper_ledger.json").write_text("{not json")
    with pytest.raises(LedgerCorrupt):
        make(tmp_path, Market())
    moved = quarantine_ledger(str(tmp_path))
    assert len(moved) == 1 and os.path.exists(moved[0])
    assert make(tmp_path, Market()).ledger["cash"] == 1000


def test_failed_orders_are_handled(tmp_path):
    m = Market()
    ex = PaperExecutor(failure_rate=0.5, seed=3)
    t = make(tmp_path, m, executor=ex)
    decisions = run(t, m, 1000)
    actions = {d["action"] for d in decisions}
    assert "BUY_FAILED" in actions or "SELL_FAILED" in actions
    assert t.ledger["failed_orders"]
    assert t.reconcile() == []


def test_failed_exit_is_retried(tmp_path):
    m = Market()
    t = make(tmp_path, m)
    open_any_position(t, m)
    pair = next(iter(t.ledger["positions"]))
    t.ledger["pending_exits"][pair] = "stop_loss"
    t.save()
    out = t.step([pair])
    assert out[0]["action"] == "SELL" and out[0]["reason"].startswith("retry:")
    assert pair not in t.ledger["positions"]


def test_emergency_stop_closes_everything_and_blocks_entries(tmp_path):
    m = Market()
    t = make(tmp_path, m)
    open_any_position(t, m)
    t.emergency_stop("test")
    assert t.ledger["positions"] == {}
    assert t.risk.kill_switch_active()
    later = run(t, m, 200)
    assert not any(d["action"] == "BUY" for d in later)
    assert t.reconcile() == []


def test_reconcile_detects_tampering(tmp_path):
    m = Market()
    t = make(tmp_path, m)
    run(t, m, 600)
    t.ledger["cash"] += 5
    assert t.reconcile()


def test_reset(tmp_path):
    m = Market()
    t = make(tmp_path, m)
    run(t, m, 300)
    t.reset(500)
    assert t.ledger["cash"] == 500 and t.ledger["trades"] == []


def test_bad_data_reported_not_raised(tmp_path):
    t = PaperTrader(PaperSettings(), str(tmp_path), data_fn=lambda *a: pd.DataFrame(), now_fn=Market().now)
    out = t.step(["X"])
    assert out[0]["action"] == "DATA_ERROR"


# ------------------------------------------------------------- executors
def test_live_trading_locked():
    assert LIVE_TRADING_LOCKED is True
    with pytest.raises(LiveTradingDisabledError):
        place_live_order("BTCUSDT", "BUY", 1, 1)
    with pytest.raises(LiveTradingDisabledError):
        get_executor("live")
    assert isinstance(get_executor("paper"), PaperExecutor)


def test_paper_executor_rules():
    ex = PaperExecutor(fee_rate=0.001, slippage_pct=0.01)
    r = ex.market_order("A", "buy", 2, 100, client_id="c1")
    assert r.ok and r.price == pytest.approx(101) and r.fee == pytest.approx(2 * 101 * 0.001)
    assert ex.market_order("A", "buy", 2, 100, client_id="c1").status == "duplicate"
    assert ex.market_order("A", "buy", 0.001, 100).status == "rejected"   # below 1 USDT
    assert ex.market_order("A", "buy", -1, 100).status == "rejected"
    assert ex.market_order("A", "short", 1, 100).status == "rejected"


def test_ledger_lock_blocks_concurrent_writers(tmp_path):
    m = Market()
    t = make(tmp_path, m)
    with ledger_lock(t.ledger_path):
        with pytest.raises(LedgerBusy):
            t.step(["SYN"])
    assert t.step(["SYN"])  # lock released


def test_second_instance_sees_first_instance_progress(tmp_path):
    m = Market()
    a, b = make(tmp_path, m), make(tmp_path, m)
    open_any_position(a, m)
    b.reload()
    assert b.ledger["positions"].keys() == a.ledger["positions"].keys()
    assert b.ledger["cash"] == a.ledger["cash"]
    b.step(["SYN"])
    assert b.reconcile() == []


def test_stale_data_blocks_entries(tmp_path):
    m = Market()
    later = lambda: m.now() + timedelta(hours=3)  # clock far ahead of the newest candle
    t = PaperTrader(make(tmp_path, m).settings, str(tmp_path / "s"), data_fn=m.klines, price_fn=m.price,
                    now_fn=later)
    for _ in range(300):
        out = t.step(["SYN"])
        m.i += 1
        assert all(d["action"] != "BUY" for d in out)
    assert any(d["action"] == "DATA_STALE" for d in t.ledger["decisions"])
