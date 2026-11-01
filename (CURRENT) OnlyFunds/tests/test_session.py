"""Long-running session management, persistence, recovery, reports and paper-only safety (offline)."""
import json
import os
import re
import stat
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from core import forward_reports as R
from core import session as S
from core.forward import ForwardPaperEngine
from core.paper_trader import LedgerBusy, PaperSettings, PaperTrader, ledger_lock
from core.risk_manager import RiskLimits
from core.selection import StrategySelector
from tests.test_forward import FakeLive, no_network, settings  # noqa: F401

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Live(FakeLive):
    def close(self):
        pass


class Clock:
    def __init__(self, live):
        self.live = live

    def __call__(self):
        return self.live.now()


def make_session(dep, live, steps_engine=None, every=60.0):
    def factory():
        return steps_engine or ForwardPaperEngine(["SYN"], settings(), dep, live, selector=StrategySelector([]),
                                                  now_fn=live.now)
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        live.i += 1          # simulated time passes one candle per step
    sess = S.ForwardSession(dep, ["SYN"], "5m", every, factory, {"test": True}, now_fn=live.now, sleep_fn=sleep)
    sess._sleeps = sleeps
    return sess


def read(path, mode="r"):
    with open(path, mode) as f:
        return f.read()


def load(path):
    return json.loads(read(path))


def jsonl(path):
    return [json.loads(x) for f in sorted(os.listdir(path)) for x in read(os.path.join(path, f)).splitlines() if x.strip()]


# --------------------------------------------------------------- lifecycle and immutability
def test_session_records_and_reports(tmp_path):
    live = Live()
    dep = str(tmp_path / "dep")
    s = make_session(dep, live)
    start = s.start()
    s.run(max_steps=20)
    sdir = os.path.join(dep, "sessions", s.session_id)
    for f in ("session_start.json", "session_end.json", "heartbeat.json", "session.log"):
        assert os.path.exists(os.path.join(sdir, f)), f
    assert not os.stat(os.path.join(sdir, "session_start.json")).st_mode & stat.S_IWRITE
    end = load(os.path.join(sdir, "session_end.json"))
    assert end["reason"] == "max_steps reached" and end["counters"]["steps"] == 20
    assert start["git"]["commit"] and start["config_fingerprint"] and start["data_source"]["credentials"] == "none"
    assert start["start_utc"].endswith("+00:00")
    assert os.path.exists(os.path.join(dep, "reports", "startup", f"{s.session_id}.md"))
    steps = jsonl(os.path.join(dep, "steps"))
    assert len(steps) == 20 and all(x["ok"] for x in steps)
    log = read(os.path.join(sdir, "session.log"))
    assert re.search(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z INFO", log)          # UTC log timestamps


def test_previous_sessions_are_never_overwritten(tmp_path):
    live = Live()
    dep = str(tmp_path / "dep")
    s1 = make_session(dep, live)
    s1.start()
    s1.run(max_steps=5)
    before = {f: read(os.path.join(dep, "sessions", s1.session_id, f), "rb")
              for f in ("session_start.json", "session_end.json")}
    s2 = make_session(dep, live)
    s2.start()
    s2.run(max_steps=5)
    assert s1.session_id != s2.session_id
    for f, b in before.items():
        assert read(os.path.join(dep, "sessions", s1.session_id, f), "rb") == b


# --------------------------------------------------------------- crash recovery
def test_crash_is_detected_and_evidence_continues(tmp_path, monkeypatch):
    live = Live()
    dep = str(tmp_path / "dep")
    s1 = make_session(dep, live)
    s1.start()
    for _ in range(30):                       # run steps, then "crash": no shutdown, no end record
        s1.one_step()
        live.i += 1
    obs_before = jsonl(os.path.join(dep, "observations"))
    monkeypatch.setattr(S, "pid_alive", lambda pid: False)
    s2 = make_session(dep, live)
    start2 = s2.start()
    s2.run(max_steps=10)
    assert start2["recovered_previous_sessions"] == [s1.session_id]
    rec = load(os.path.join(dep, "sessions", s1.session_id, "session_end_recovered.json"))
    assert "unexpectedly" in rec["ended"] and rec["last_heartbeat_utc"]
    incidents = [load(os.path.join(dep, "incidents", f)) for f in os.listdir(os.path.join(dep, "incidents"))]
    assert any(i["kind"] == "unexpected_termination" for i in incidents)
    obs = jsonl(os.path.join(dep, "observations"))
    keys = [(o["pair"], o["candle"]["created_at"]) for o in obs]
    assert len(keys) == len(set(keys)) and len(obs) > len(obs_before)


def test_second_concurrent_session_refused(tmp_path):
    live = Live()
    dep = str(tmp_path / "dep")
    s1 = make_session(dep, live)
    s1.start()                                       # still "running" (this test process is alive)
    s2 = make_session(dep, live)
    with pytest.raises(RuntimeError):
        s2.start()


def test_crashed_lock_holder_released_immediately(tmp_path, monkeypatch):
    path = str(tmp_path / "ledger.json")
    with open(path + ".lock", "w") as f:
        f.write("999999")                            # pid of a process that no longer exists
    import core.paper_trader as pt
    monkeypatch.setattr(pt, "pid_alive", lambda pid: False)
    with ledger_lock(path):
        pass
    monkeypatch.setattr(pt, "pid_alive", lambda pid: True)
    with open(path + ".lock", "w") as f:
        f.write("999999")
    with pytest.raises(LedgerBusy):
        with ledger_lock(path):
            pass


# --------------------------------------------------------------- stop, outages, back-off
def test_stop_file_stops_gracefully(tmp_path):
    live = Live()
    dep = str(tmp_path / "dep")
    s = make_session(dep, live)
    s.start()
    open(os.path.join(dep, "STOP"), "w").write("stop")
    s.run(max_steps=1000)
    end = load(os.path.join(dep, "sessions", s.session_id, "session_end.json"))
    assert end["reason"] == "STOP file present"


class FailingEngine:
    def __init__(self, live, fail_steps):
        self.live, self.fail_steps, self.n = live, fail_steps, 0
        self.selector = StrategySelector([])

    def step(self):
        self.n += 1
        if self.n <= self.fail_steps:
            raise ConnectionError("network down")
        return {"observed": ["SYN"], "errors": [], "trader": [], "shadow_resolved": []}


def test_outage_backoff_incidents_and_exceptions_surfaced(tmp_path):
    live = Live()
    dep = str(tmp_path / "dep")
    s = make_session(dep, live, steps_engine=FailingEngine(live, fail_steps=8), every=60)
    s.start()                                         # first (failing) step happens in the startup check
    s.run(max_steps=12)
    delays = s._sleeps
    assert delays[:6] == [240, 480, 900, 900, 900, 900] or max(delays) == 900
    assert max(delays) <= 900 and delays[-1] == 60   # bounded, and back to normal after recovery
    kinds = [load(os.path.join(dep, "incidents", f))["kind"] for f in os.listdir(os.path.join(dep, "incidents"))]
    assert "data_outage" in kinds and "outage_recovered" in kinds
    assert s.counters["exceptions"] == 8
    steps = jsonl(os.path.join(dep, "steps"))
    assert sum(1 for x in steps if not x["ok"] and "network down" in x["errors"][0]) == 8
    assert "network down" in read(os.path.join(dep, "sessions", s.session_id, "session.log"))


# --------------------------------------------------------------- reports
def test_daily_report_written_once_at_utc_rollover(tmp_path):
    live = Live(n=3000, i=600)
    dep = str(tmp_path / "dep")
    s = make_session(dep, live)
    s.start()
    s.run(max_steps=320)                              # 5m candles: crosses at least one UTC midnight
    daily = os.listdir(os.path.join(dep, "reports", "daily"))
    assert daily
    p = os.path.join(dep, "reports", "daily", daily[0])
    before = read(p, "rb")
    assert R.write_report(dep, "daily", daily[0][:-3], "something else") is None    # never overwritten
    assert read(p, "rb") == before
    text = before.decode()
    for section in ("## Market data", "## Paper trading (simulated)", "## Shadow tracking"):
        assert section in text


def test_evidence_wording():
    assert R.evidence_status(pd.Series([-1.0] * 40))["status"] == "INSUFFICIENT"
    assert R.evidence_status(pd.Series([-1.0, 0.2] * 100))["status"] == "NEGATIVE"
    assert R.evidence_status(pd.Series([-1.0, 1.0] * 100))["status"] == "INCONCLUSIVE"
    assert R.evidence_status(pd.Series([2.0, -0.5] * 100))["status"].startswith("POSITIVE (forward sample, not a validation)")


def test_trial_report_checks(tmp_path, monkeypatch):
    from core.trial import trial_report
    live = Live()
    dep = str(tmp_path / "dep")
    s1 = make_session(dep, live)
    s1.start()
    for _ in range(20):
        s1.one_step()
        live.i += 1
    monkeypatch.setattr(S, "pid_alive", lambda pid: False)
    s2 = make_session(dep, live)
    s2.start()
    s2.run(max_steps=20)
    text = trial_report(dep, None, 60.0)
    for k in ("records_persisted", "no_duplicates", "recovery_tested", "no_live_orders_possible",
              "no_positions_without_approval"):
        assert f'"{k}": true' in text, k
    assert "INSUFFICIENT" in text


# --------------------------------------------------------------- ledger archive (no evidence deleted)
def test_ledger_overflow_is_archived_not_deleted(tmp_path, monkeypatch):
    import core.paper_trader as pt
    monkeypatch.setattr(pt, "MAX_DECISIONS", 50)
    t = PaperTrader(PaperSettings(risk=RiskLimits(kill_switch_file="")), str(tmp_path), data_fn=lambda *a: None)
    for k in range(180):
        t._decide("X", "WAIT", f"n{k}")
        if k % 30 == 0:
            t.save()
    t.save()
    archived = [json.loads(x) for x in read(tmp_path / "archive" / "decisions.jsonl").splitlines()]
    reasons = [d["reason"] for d in archived] + [d["reason"] for d in t.ledger["decisions"]]
    assert reasons == [f"n{k}" for k in range(180)]
    assert t.ledger["archived_counts"]["decisions"] == 130


# --------------------------------------------------------------- paper-only safety (static and dynamic)
FORWARD_FILES = ["core/forward.py", "core/live_data.py", "core/session.py", "core/forward_reports.py", "core/trial.py",
                 "core/selection.py", "core/regime.py", "core/monitoring.py", "run_forward_session.py",
                 "run_forward_paper.py", "scripts/forward_supervisor.py"]


def test_forward_path_has_no_order_capability():
    for f in FORWARD_FILES:
        with open(os.path.join(APP, f), encoding="utf-8") as fh:
            src = fh.read()
        for bad in ("place_order", "cancel_order", "allow_trading=True", "CoinExClient", "COINEX_API_KEY",
                    "COINEX_API_SECRET", ".post(", "requests.post", "X-MBX-APIKEY", "signature"):
            assert bad not in src, f"{bad} found in {f}"


def test_allowlist_excludes_order_and_account_endpoints():
    from core.live_data import ALLOWED_PATHS
    assert all(("order" not in p and "account" not in p and "withdraw" not in p) for p in ALLOWED_PATHS)


def test_forward_session_never_promotes_strategies(tmp_path):
    from core.run_store import DEFAULT_ROOT
    before = sorted(os.walk(DEFAULT_ROOT))
    live = Live()
    s = make_session(str(tmp_path / "dep"), live)
    s.start()
    s.run(max_steps=30)
    assert sorted(os.walk(DEFAULT_ROOT)) == before          # forward evidence never writes validation runs
    assert StrategySelector().approvals == []
    led = load(tmp_path / "dep" / "paper" / "paper_ledger.json")
    assert led["positions"] == {} and led["trades"] == [] and led["cash"] == led["initial_capital"]
