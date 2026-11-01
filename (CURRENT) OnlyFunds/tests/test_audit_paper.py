"""Second-pass audit: attempts to break the paper trader."""
import json
import os
import threading
import time
from datetime import timedelta

import pandas as pd
import pytest
import requests

from core.paper_trader import LedgerBusy, LedgerCorrupt, PaperSettings, PaperTrader, ledger_lock
from core.positions import ExitRules, Position
from core.risk_manager import RiskLimits
from core.trade_execution import PaperExecutor
from tests.test_paper_trader import Market, make, no_network, open_any_position, run  # noqa: F401


# ------------------------------------------------------------------ ledger durability
def test_corrupt_ledger_recovers_from_backup(tmp_path):
    m = Market()
    t = make(tmp_path, m)
    open_any_position(t, m)
    run(t, m, 5)                                  # several saves -> a .bak exists
    good = json.loads((tmp_path / "paper_ledger.json.bak").read_text())
    (tmp_path / "paper_ledger.json").write_text("{truncated")
    t2 = make(tmp_path, m)
    assert t2.ledger["cash"] == good["cash"] and t2.ledger["positions"].keys() == good["positions"].keys()


def test_corrupt_ledger_without_backup_refuses_to_start(tmp_path):
    (tmp_path / "paper_ledger.json").write_text("{not json")
    with pytest.raises(LedgerCorrupt):
        make(tmp_path, Market())
    assert (tmp_path / "paper_ledger.json").exists(), "corrupt ledger must be kept for inspection"


def test_crash_between_backup_and_write_is_recoverable(tmp_path):
    m = Market()
    t = make(tmp_path, m)
    run(t, m, 3)
    os.replace(tmp_path / "paper_ledger.json", tmp_path / "paper_ledger.json.bak")  # simulated crash point
    t2 = make(tmp_path, m)
    assert len(t2.ledger["equity_history"]) == 3


def test_structurally_invalid_ledger_rejected(tmp_path):
    (tmp_path / "paper_ledger.json").write_text(json.dumps({"cash": "lots", "positions": []}))
    with pytest.raises(LedgerCorrupt):
        make(tmp_path, Market())


def test_inconsistent_ledger_blocks_new_entries(tmp_path):
    m = Market()
    t = make(tmp_path, m)
    run(t, m, 20)
    t.ledger["cash"] += 500                       # money appears from nowhere
    t.ledger["positions"] = {}
    t.save()
    out = run(t, m, 600)
    assert not any(d["action"] == "BUY" for d in out)
    assert any(d["action"] == "BLOCKED" and "reconcil" in d["reason"] for d in out)


# ------------------------------------------------------------------ execution realism
def test_stop_found_in_replay_fills_at_current_price_if_worse(tmp_path):
    m = Market()
    t = make(tmp_path, m)
    open_any_position(t, m)
    pair = next(iter(t.ledger["positions"]))
    pos = t.positions[pair]
    m.i += 1
    # Next completed bar wicks through the stop; current price is far below it.
    m.df.loc[m.i, ["Open", "High", "Low", "Close"]] = [pos.entry_price, pos.entry_price, pos.stop_loss * 0.99,
                                                        pos.stop_loss * 0.95]
    m.df.loc[m.i + 1, "Open"] = pos.stop_loss * 0.90
    out = t.step([pair])
    sell = [d for d in out if d["action"] == "SELL"][0]
    assert sell["price"] <= pos.stop_loss * 0.90 + 1e-9, "fill must not be better than the market at execution"


def test_entry_candle_is_checked_for_stops(tmp_path):
    m = Market()
    t = make(tmp_path, m)
    open_any_position(t, m)                       # entered at the open of bar m.i (ticker = next open)
    pair = next(iter(t.ledger["positions"]))
    pos = t.positions[pair]
    # The entry candle (now index m.i) dips through the stop and recovers before it closes.
    m.df.loc[m.i, ["High", "Low", "Close"]] = [pos.entry_price * 1.001, pos.stop_loss * 0.995, pos.entry_price]
    m.df.loc[m.i + 1, "Open"] = pos.entry_price
    m.i += 1
    out = t.step([pair])
    assert any(d["action"] == "SELL" and "stop" in d["reason"] for d in out)


def test_stop_exit_uses_stop_slippage(tmp_path):
    m = Market()
    ex = PaperExecutor(fee_rate=0.002, slippage_pct=0.0005)
    t = make(tmp_path, m, executor=ex)
    t.settings.stop_slippage_pct = 0.004
    open_any_position(t, m)
    pair = next(iter(t.ledger["positions"]))
    pos = t.positions[pair]
    d = t._sell(pos, 100.0, "stop_loss")
    assert d["price"] == pytest.approx(100.0 * (1 - 0.004))


def test_paper_sizing_budgets_costs_and_liquidity(tmp_path):
    m = Market()
    m.df["Volume"] = 0.5                          # tiny volume: 10% participation = 0.05 units
    t = make(tmp_path, m)
    for _ in range(1500):
        out = t.step(["SYN"])
        m.i += 1
        buys = [d for d in out if d["action"] == "BUY"]
        if buys:
            assert buys[0]["qty"] <= 0.05 + 1e-12
            return
    pytest.fail("no entry happened")


# ------------------------------------------------------------------ data problems
def test_missing_candles_block_entries(tmp_path):
    m = Market()
    gappy = m.df.drop(index=range(900, 1100, 7)).reset_index(drop=True)
    m.df = gappy
    t = make(tmp_path, m)
    m.i = 950
    out = run(t, m, 100)
    assert not any(d["action"] == "BUY" for d in out)
    assert any(d["action"] == "DATA_GAP" for d in out)


def test_network_interruption_then_recovery(tmp_path):
    m = Market()
    t = make(tmp_path, m)
    open_any_position(t, m)
    down = {"on": True}
    real = t.data_fn

    def flaky(pair, interval, limit):
        if down["on"]:
            raise requests.ConnectionError("network down")
        return real(pair, interval, limit)

    t.data_fn = flaky
    t.price_fn = lambda p: (_ for _ in ()).throw(requests.ConnectionError("down")) if down["on"] else m.price(p)
    before = dict(t.ledger["positions"])
    outs = run(t, m, 5)
    assert all(d["action"] == "ERROR" for d in outs)
    assert t.ledger["positions"].keys() == before.keys()
    assert t.reconcile() == []
    down["on"] = False
    run(t, m, 50)
    assert t.reconcile() == []


# ------------------------------------------------------------------ concurrency
def test_app_and_runner_concurrently(tmp_path):
    """Two trader instances (UI + run_paper.py) hammering the same ledger from threads."""
    m = Market()
    lock = threading.Lock()
    a, b = make(tmp_path, m), make(tmp_path, m)
    busy = {"a": 0, "b": 0}
    errors = []

    def worker(tr, name, n):
        for _ in range(n):
            try:
                tr.step(["SYN"])
            except LedgerBusy:
                busy[name] += 1
            except Exception as exc:  # pragma: no cover - surfaced below
                errors.append(exc)
            with lock:
                m.i += 1

    th = [threading.Thread(target=worker, args=(a, "a", 150)), threading.Thread(target=worker, args=(b, "b", 150))]
    [x.start() for x in th]
    [x.join() for x in th]
    assert not errors
    final = make(tmp_path, m)
    assert final.reconcile() == []
    ids = [tr["client_id"] for tr in final.ledger["trades"]]
    assert len(ids) == len(set(ids))
    assert len(final.ledger["positions"]) <= 1


def test_long_step_keeps_lock_alive(tmp_path, monkeypatch):
    import core.paper_trader as pt
    monkeypatch.setattr(pt, "LOCK_STALE_SECONDS", 0.3)
    m = Market()
    slow = make(tmp_path, m)
    real = slow.data_fn

    def slow_data(*a):
        time.sleep(0.2)
        return real(*a)

    slow.data_fn = slow_data
    result = {}

    def other():
        time.sleep(0.5)
        try:
            with ledger_lock(slow.ledger_path, stale_after=0.3):
                result["stole"] = True
        except LedgerBusy:
            result["stole"] = False

    th = threading.Thread(target=other)
    th.start()
    slow.step(["SYN"] * 6)                        # ~1.2 s, longer than the stale limit
    th.join()
    assert result["stole"] is False


# ------------------------------------------------------------------ duplicates / emergency
def test_duplicate_order_ids_rejected_after_restart(tmp_path):
    m = Market()
    t = make(tmp_path, m)
    open_any_position(t, m)
    cid = t.ledger["client_ids"][-1]
    t2 = make(tmp_path, m)
    assert t2.executor.market_order("SYN", "buy", 1, 100, client_id=cid).status == "duplicate"


def test_emergency_stop_survives_restart(tmp_path):
    m = Market()
    t = make(tmp_path, m)
    open_any_position(t, m)
    t.emergency_stop("drill")
    t2 = make(tmp_path, m)
    assert t2.ledger["positions"] == {} and t2.risk.kill_switch_active()
    assert not any(d["action"] == "BUY" for d in run(t2, m, 300))


def test_lock_released_between_checks_reports_busy_not_crash(tmp_path, monkeypatch):
    """Regression: the holder deleting the lock between our open() and getmtime() raised FileNotFoundError."""
    path = str(tmp_path / "paper_ledger.json")
    open(path + ".lock", "w").close()
    import core.paper_trader as pt

    def vanished(_):
        raise FileNotFoundError("lock released")

    monkeypatch.setattr(pt.os.path, "getmtime", vanished)
    with pytest.raises(LedgerBusy):
        with ledger_lock(path):
            pass


def test_lock_stress_two_threads(tmp_path):
    path = str(tmp_path / "paper_ledger.json")
    errors, held, got = [], {"n": 0}, {"n": 0}
    guard = threading.Lock()

    def hammer():
        for _ in range(3000):
            try:
                with ledger_lock(path):
                    with guard:
                        held["n"] += 1
                        assert held["n"] == 1, "two holders at once"
                        got["n"] += 1
                    with guard:
                        held["n"] -= 1
            except LedgerBusy:
                pass
            except Exception as exc:
                errors.append(repr(exc))

    th = [threading.Thread(target=hammer) for _ in range(2)]
    [t.start() for t in th]
    [t.join() for t in th]
    assert errors == [] and got["n"] > 0
