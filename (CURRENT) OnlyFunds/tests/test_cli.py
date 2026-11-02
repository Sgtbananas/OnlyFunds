"""CLI entry points with the exchange faked (no network)."""
import importlib
import json
import os
import subprocess
import sys

import pytest
import requests

from core import coinex_client as cx
from core import core_data
from tests.synthetic import synthetic_ohlcv

APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = {p: synthetic_ohlcv(1200, seed=i + 3, start=50 * (i + 1)) for i, p in enumerate(["BTCUSDT", "ETHUSDT"])}


@pytest.fixture
def fake_exchange(tmp_path, monkeypatch):
    monkeypatch.setenv("OF2_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("OF2_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(requests.Session, "request",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("network used")))
    monkeypatch.setattr(cx.CoinExClient, "get_markets", lambda self, refresh=False, ttl=600: {})
    monkeypatch.setattr(cx.CoinExClient, "get_ticker",
                        lambda self, m: {"last": str(DATA[m]["Close"].iloc[-1])})
    monkeypatch.setattr(cx.CoinExClient, "get_klines",
                        lambda self, m, interval="5m", limit=500, **k: DATA[m].tail(limit).reset_index(drop=True))
    core_data._client = None
    return tmp_path


def test_run_paper_two_steps(fake_exchange):
    import run_paper
    importlib.reload(run_paper)  # pick up OF2_STATE_DIR
    assert run_paper.main(["--pairs", "BTCUSDT", "ETHUSDT", "--steps", "2", "--every", "0"]) == 0
    ledger = json.loads((fake_exchange / "state" / "paper" / "paper_ledger.json").read_text())
    assert len(ledger["equity_history"]) == 2
    assert {d["pair"] for d in ledger["decisions"]} == {"BTCUSDT", "ETHUSDT"}


def test_backtest_suite_and_auto_tuner(fake_exchange, monkeypatch, capsys):
    from scripts import auto_tuner, backtest_suite
    import utils.config as uc
    monkeypatch.setattr(uc, "APP_ROOT", str(fake_exchange))
    monkeypatch.setattr(backtest_suite, "APP_ROOT", str(fake_exchange))
    monkeypatch.setattr(auto_tuner, "APP_ROOT", str(fake_exchange))
    assert backtest_suite.main(["--pairs", "BTCUSDT", "--oos"]) == 0
    assert "out_of_sample" in capsys.readouterr().out
    assert (fake_exchange / "state" / "backtest_results.json").exists()
    assert auto_tuner.main(["--pairs", "ETHUSDT"]) == 0
    rep = json.loads((fake_exchange / "state" / "auto_params.json").read_text())
    assert isinstance(rep["ETHUSDT"]["confirmed"], bool)


def test_run_live_refuses():
    r = subprocess.run([sys.executable, "run_live.py"], cwd=APP_ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "locked" in r.stdout.lower()
