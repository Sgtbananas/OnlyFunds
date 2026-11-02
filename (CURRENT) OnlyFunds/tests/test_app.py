"""Streamlit integration test: runs main.py headless with CoinEx faked (no network)."""
import os

import pytest
import requests
from streamlit.testing.v1 import AppTest

from core import coinex_client as cx
from core import core_data
from tests.synthetic import synthetic_ohlcv

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main.py")

MKT = lambda m, base: cx.MarketInfo(m, base, "USDT", 6, 2, 0.0001, 0.002, 0.002, "online", True)
MARKETS = {"BTCUSDT": MKT("BTCUSDT", "BTC"), "ETHUSDT": MKT("ETHUSDT", "ETH"), "SOLUSDT": MKT("SOLUSDT", "SOL")}
DATA = {p: synthetic_ohlcv(1200, seed=i + 1, start=100 * (i + 1)) for i, p in enumerate(MARKETS)}


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OF2_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("OF2_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(requests.Session, "request",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("network used in UI test")))
    monkeypatch.setattr(cx.CoinExClient, "get_markets", lambda self, refresh=False, ttl=600: MARKETS)
    monkeypatch.setattr(cx.CoinExClient, "get_tickers",
                        lambda self, markets=None: [{"market": m, "value": str(1000 - i)} for i, m in enumerate(MARKETS)])
    monkeypatch.setattr(cx.CoinExClient, "get_ticker",
                        lambda self, m: {"market": m, "last": str(DATA[m]["Close"].iloc[-1])})
    monkeypatch.setattr(cx.CoinExClient, "get_klines",
                        lambda self, m, interval="5m", limit=500, **k: DATA[m].tail(limit).reset_index(drop=True))
    core_data._client = None
    yield tmp_path


def run_app():
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def button(at, label):
    for b in at.button:
        if b.label.startswith(label):
            return b
    raise AssertionError(f"button {label!r} not found; have {[b.label for b in at.button]}")


def test_app_starts_and_shows_market(app_env):
    at = run_app()
    assert at.title[0].value.startswith("📈 OnlyFunds 2")
    assert any("Paper trading only" in s.value for s in at.sidebar.success)
    labels = [m.label for m in at.metric]
    assert {"Last close", "Signal", "Regime", "Decision"} <= set(labels)
    assert not at.error, [e.value for e in at.error]


def test_backtest_from_ui(app_env):
    at = run_app()
    button(at, "▶ Run backtest").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert any(m.label == "Win rate" for m in at.metric)
    assert os.path.exists(os.path.join(app_env, "state", "backtest_results.json"))


def test_paper_step_from_ui(app_env):
    at = run_app()
    button(at, "▶ Run one step now").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert os.path.exists(os.path.join(app_env, "state", "paper", "paper_ledger.json"))
    assert any(m.label == "Equity" for m in at.metric)


def test_status_controls(app_env):
    at = run_app()
    button(at, "Reconcile paper ledger").click().run()
    assert any("Ledger consistent" in s.value for s in at.success)
    button(at, "Test CoinEx connection").click().run()
    assert any("CoinEx public API OK" in s.value for s in at.success)
    for cb in at.checkbox:
        if cb.label == "Confirm emergency stop":
            cb.check()
    at.run()
    button(at, "🛑 Emergency stop").click().run()
    assert os.path.exists(os.path.join(app_env, "state", "KILL_SWITCH"))
    button(at, "Clear kill switch").click().run()
    assert not os.path.exists(os.path.join(app_env, "state", "KILL_SWITCH"))


def test_optimisation_and_ml_from_ui(app_env):
    at = run_app()
    button(at, "Run optimisation").click().run()
    assert not at.exception, [e.value for e in at.exception]
    button(at, "Train ML filter").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert os.path.exists(os.path.join(app_env, "state", "ml_filter.pkl"))


def test_corrupt_ledger_does_not_take_down_other_tabs(app_env):
    paper = app_env / "state" / "paper"
    paper.mkdir(parents=True)
    (paper / "paper_ledger.json").write_text("{broken")
    at = run_app()
    assert any("cannot be read" in e.value for e in at.error)
    assert any(b.label == "Test CoinEx connection" for b in at.button), "status tab must still render"
    assert (paper / "paper_ledger.json").exists()


def test_backtest_shows_new_metrics(app_env):
    at = run_app()
    button(at, "▶ Run backtest").click().run()
    labels = {m.label for m in at.metric}
    assert {"Expectancy / trade", "Trades / day", "Sortino"} <= labels
