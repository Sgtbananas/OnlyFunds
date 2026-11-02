"""Scheduled-task wrapper for the shadow tracker: lock, journal, crash records, credential isolation."""
import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "research"))
import shadow_task as ST  # noqa: E402
from tests.test_forward_shadow import shadow  # noqa: E402,F401  (offline fixture: synthetic data, network blocked)


def journal(sd):
    with open(os.path.join(sd, "task_journal.jsonl"), encoding="utf-8") as f:
        return [json.loads(x) for x in f]


@pytest.fixture
def sd(tmp_path, monkeypatch):
    monkeypatch.setenv("OF2_STATE_DIR", str(tmp_path))
    return os.path.join(str(tmp_path), "shadow")


def test_second_instance_is_refused(sd):
    lock = ST.InstanceLock(os.path.join(sd, "tracker.lock"))
    assert lock.acquire()
    try:
        assert ST.run(tracker_main=lambda a: 0) == ST.EXIT_LOCKED
    finally:
        lock.release()
    assert journal(sd)[-1]["status"].startswith("REFUSED")
    assert ST.run(tracker_main=lambda a: 0) == 0                       # free again after release


def test_crash_is_journaled_with_traceback(sd):
    def boom(argv):
        os.makedirs(os.path.join(sd, "runs", "20990101T000000Z"))
        raise ValueError("synthetic failure")
    assert ST.run(tracker_main=boom) == ST.EXIT_CRASHED
    e = journal(sd)[-1]
    assert e["status"].startswith("CRASHED") and "synthetic failure" in e["traceback"]
    assert os.path.exists(e["log"]) and "synthetic failure" in open(e["log"], encoding="utf-8").read()
    assert ST.InstanceLock(os.path.join(sd, "tracker.lock")).acquire()  # lock released after a crash


def test_credentials_are_removed_before_the_tracker_runs(sd, monkeypatch):
    monkeypatch.setenv("COINEX_API_KEY", "x")
    monkeypatch.setenv("COINEX_API_SECRET", "y")
    seen = {}
    ST.run(tracker_main=lambda a: seen.update({k: os.environ.get(k) for k in ("COINEX_API_KEY", "COINEX_API_SECRET")}))
    assert seen == {"COINEX_API_KEY": None, "COINEX_API_SECRET": None}
    assert set(journal(sd)[-1]["credential_env_removed"]) == {"COINEX_API_KEY", "COINEX_API_SECRET"}


def test_full_update_through_wrapper_never_touches_the_exchange(shadow, monkeypatch, tmp_path):  # noqa: F811
    import forward_shadow as FS
    import core.coinex_client as CX
    monkeypatch.setattr(CX.CoinExClient, "__init__", lambda *a, **k: (_ for _ in ()).throw(AssertionError("exchange")))
    monkeypatch.setattr(ST, "state_dir", lambda: FS.STATE)
    FS.cmd_freeze()
    assert ST.run(tracker_main=lambda a: FS.cmd_update(skip_random=True)) == 0
    e = journal(FS.STATE)[-1]
    assert e["status"] == "ok" and e["anomalies"] == [] and len(e["runs"]) == 1
    assert any("EOSUSDT" in w for w in e["data_warnings"])            # unavailable coin is reported, not hidden


def test_wrapper_has_no_order_paths():
    src = open(os.path.join(ROOT, "research", "shadow_task.py"), encoding="utf-8").read()
    for bad in ("place_order", "CoinExClient", "requests", "allow_trading", "LIVE_TRADING_LOCKED ="):
        assert bad not in src
