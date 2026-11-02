"""Shadow alerting and integrity: one notification per new issue, severity, escalation, resolution, exit codes, the
ledger checkpoint (what it detects and what it cannot), and an opt-in real Windows notification. Offline."""
import json
import os
import shutil
import sys
from datetime import datetime, timezone

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "research"))
import shadow_health as SH  # noqa: E402
import shadow_task as ST  # noqa: E402
from tests.test_forward_shadow import shadow  # noqa: E402,F401  (offline fixture: synthetic data, network blocked)
from tests.test_shadow_monitoring import NOW, _days, _state  # noqa: E402


class Notifier:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def __call__(self, title, line1, body, level, tag):
        if self.fail:
            raise OSError("no session")
        self.calls.append({"title": title, "line1": line1, "body": body, "level": level, "tag": tag})
        return {"method": "fake", "delivered": True}


def _jr(sd):
    with open(os.path.join(sd, "task_journal.jsonl"), encoding="utf-8") as f:
        return [json.loads(x) for x in f]


def _alerts(sd):
    p = os.path.join(sd, "alerts.jsonl")
    return [json.loads(x) for x in open(p, encoding="utf-8")] if os.path.exists(p) else []


def _issues(*items):
    return lambda root, now, entry: {"ok": not items, "checked_utc": now.isoformat(), "findings": [m for _, _, m in items],
                                     "issues": [{"key": k, "level": lv, "message": m} for k, lv, m in items]}


@pytest.fixture
def sd(tmp_path, monkeypatch):
    monkeypatch.setenv("OF2_STATE_DIR", str(tmp_path))
    return os.path.join(str(tmp_path), "shadow")


# ------------------------------------------------------------------ alerting
def test_critical_issue_notifies_once_and_keeps_exit_11_until_resolved(sd):
    n = Notifier()
    hc = _issues(("missed:2026-11-04", "critical", "missed runs: no successful tracker run on UTC day 2026-11-04"))
    assert ST.run(tracker_main=lambda a: 0, health_check=hc, notifier=n) == ST.EXIT_HEALTH_ALERT
    assert len(n.calls) == 1
    c = n.calls[0]
    assert c["level"] == "critical" and "CRITICAL" in c["title"]
    assert "2026-11-04" in c["body"] and "missed runs" in c["body"]                      # the failed check
    assert datetime.now(timezone.utc).strftime("%Y-%m-%d") in c["line1"] and "UTC" in c["line1"]   # date and time
    a = _alerts(sd)
    assert [x["event"] for x in a] == ["raised"] and a[0]["notification"]["delivered"]
    md = open(os.path.join(sd, "HEALTH_LATEST.md"), encoding="utf-8").read()
    assert md.startswith("# OF2 shadow health: CRITICAL") and "missed runs" in md
    # same unresolved issue: no second notification, no second record, still exit 11
    assert ST.run(tracker_main=lambda a: 0, health_check=hc, notifier=n) == ST.EXIT_HEALTH_ALERT
    assert len(n.calls) == 1 and len(_alerts(sd)) == 1
    assert _jr(sd)[-1]["health"]["new"] == [] and _jr(sd)[-1]["health"]["critical_open"] == 1
    # resolved: recorded, exit 0, nothing notified
    assert ST.run(tracker_main=lambda a: 0, health_check=_issues(), notifier=n) == 0
    assert [x["event"] for x in _alerts(sd)] == ["raised", "resolved"] and len(n.calls) == 1
    assert json.load(open(os.path.join(sd, "alert_state.json")))["open"] == {}
    assert open(os.path.join(sd, "HEALTH_LATEST.md"), encoding="utf-8").read().startswith("# OF2 shadow health: OK")


def test_warning_is_distinguished_and_does_not_fail_the_task(sd):
    n = Notifier()
    hc = _issues(("evaluation_lag", "warning", "evaluation lags: run on 2026-11-05 evaluated only through 2026-11-02"))
    assert ST.run(tracker_main=lambda a: 0, health_check=hc, notifier=n) == 0
    assert n.calls[0]["level"] == "warning" and "WARNING" in n.calls[0]["title"]
    assert _alerts(sd)[0]["level"] == "warning"
    assert open(os.path.join(sd, "HEALTH_LATEST.md"), encoding="utf-8").read().startswith("# OF2 shadow health: WARNING")


def test_escalation_from_warning_to_critical_notifies_again(sd):
    n = Notifier()
    ST.run(tracker_main=lambda a: 0, health_check=_issues(("k", "warning", "something odd")), notifier=n)
    assert ST.run(tracker_main=lambda a: 0, health_check=_issues(("k", "critical", "now broken")), notifier=n) == 11
    assert [c["level"] for c in n.calls] == ["warning", "critical"]
    assert [x["event"] for x in _alerts(sd)] == ["raised", "escalated"]


def test_a_second_new_issue_notifies_while_the_first_stays_quiet(sd):
    n = Notifier()
    a = ("a", "critical", "first problem")
    ST.run(tracker_main=lambda x: 0, health_check=_issues(a), notifier=n)
    ST.run(tracker_main=lambda x: 0, health_check=_issues(a, ("b", "critical", "second problem")), notifier=n)
    assert len(n.calls) == 2 and "second problem" in n.calls[1]["body"] and "first problem" not in n.calls[1]["body"]


def test_notifier_failure_keeps_the_alert_record(sd):
    hc = _issues(("x", "critical", "snapshot BTCUSDT does not match its recorded hash"))
    assert ST.run(tracker_main=lambda a: 0, health_check=hc, notifier=Notifier(fail=True)) == ST.EXIT_HEALTH_ALERT
    a = _alerts(sd)[0]
    assert a["event"] == "raised" and not a["notification"]["delivered"] and "notifier failed" in a["notification"]["detail"]


def test_health_check_crash_is_a_critical_issue(sd):
    def broken(root, now, entry):
        raise RuntimeError("health exploded")
    n = Notifier()
    assert ST.run(tracker_main=lambda a: 0, health_check=broken, notifier=n) == ST.EXIT_HEALTH_ALERT
    assert "health check crashed" in n.calls[0]["body"] and _jr(sd)[-1]["exit_code"] == 0
    assert ST.InstanceLock(os.path.join(sd, "tracker.lock")).acquire()


def test_failed_tracker_keeps_its_exit_code_and_is_reported(sd):
    n = Notifier()
    hc = lambda root, now, entry: SH.check(root, now=now, pending_entry=entry)
    assert ST.run(tracker_main=lambda a: 5, health_check=hc, notifier=n) == 5
    assert n.calls[0]["level"] == "critical" and "journal: exit 5" in n.calls[0]["body"]


def test_wrapper_failure_after_the_tracker_is_journaled(sd, monkeypatch):
    monkeypatch.setattr(ST, "_gap_warnings", lambda rec: 1 / 0)

    def tracker(argv):
        d = os.path.join(sd, "runs", "20990101T000000Z")
        os.makedirs(d)
        with open(os.path.join(d, "run.json"), "w") as f:
            json.dump({"status": "ok"}, f)
        return 0
    assert ST.run(tracker_main=tracker) == ST.EXIT_CRASHED
    assert _jr(sd)[-1]["status"].startswith("CRASHED: wrapper error")
    assert ST.InstanceLock(os.path.join(sd, "tracker.lock")).acquire()


def test_full_update_with_production_health_and_checkpoint(shadow, monkeypatch, tmp_path):  # noqa: F811
    import forward_shadow as FS
    cdir = str(tmp_path / "outside_checkpoint")
    monkeypatch.setenv("OF2_CHECKPOINT_DIR", cdir)
    monkeypatch.setattr(ST, "state_dir", lambda: FS.STATE)
    FS.cmd_freeze()
    n = Notifier()
    upd = lambda a: FS.cmd_update(skip_random=True)
    code = ST.run(tracker_main=upd, health_check=ST._default_health, notifier=n, checkpoint=ST._default_checkpoint)
    e = _jr(FS.STATE)[-1]
    rep = json.load(open(os.path.join(FS.STATE, "HEALTH_LATEST.json"), encoding="utf-8"))
    assert e["status"] == "ok" and rep["ok"], rep["findings"]
    assert code == 0 and n.calls == []
    ledger = [json.loads(x) for x in open(os.path.join(FS.STATE, SH.LEDGER), encoding="utf-8")]
    assert [x["run_id"] for x in ledger] == e["runs"] and not ledger[0]["backfilled"]
    assert os.path.exists(os.path.join(cdir, SH.CHECKPOINT))
    # the whole ledger is deleted: caught through the checkpoint, raised as critical, notified once
    os.remove(os.path.join(FS.STATE, SH.LEDGER))
    assert ST.run(tracker_main=upd, health_check=ST._default_health, notifier=n,
                  checkpoint=ST._default_checkpoint) == ST.EXIT_HEALTH_ALERT
    assert len(n.calls) == 1 and "run ledger deleted" in n.calls[0]["body"]
    assert not os.path.exists(os.path.join(FS.STATE, SH.LEDGER))          # not silently rebuilt


# ------------------------------------------------------------------ integrity guarantees (and their limits)
def _cp_state(tmp_path):
    run = _state(tmp_path, _days("2026-11-05"))
    second = tmp_path / "shadow" / "runs" / "20261104T003000Z"
    second.mkdir(parents=True)
    (second / "run.json").write_text(json.dumps({"status": "waiting"}))
    cdir = str(tmp_path / "cp")
    SH.ledger_append(str(tmp_path), now=NOW, checkpoint_dir=cdir)
    assert SH.update_checkpoint(str(tmp_path), cdir, now=NOW) == []
    return run, cdir


def _cp_text(tmp_path, cdir):
    return " | ".join(SH.check(str(tmp_path), now=NOW, checkpoint_dir=cdir)["findings"])


def test_consistent_state_with_checkpoint_passes(tmp_path):
    _, cdir = _cp_state(tmp_path)
    rep = SH.check(str(tmp_path), now=NOW, checkpoint_dir=cdir)
    assert rep["ok"], rep["findings"]


def test_record_modification_and_deletion_are_detected(tmp_path):
    run, cdir = _cp_state(tmp_path)
    (run / "data" / "BTCUSDT_1d.csv").write_bytes(b"x")
    assert "changed after it was recorded" in _cp_text(tmp_path, cdir)
    shutil.rmtree(run)
    assert "was deleted" in _cp_text(tmp_path, cdir)


def test_reordering_ledger_entries_is_detected(tmp_path):
    _, cdir = _cp_state(tmp_path)
    lp = tmp_path / "shadow" / SH.LEDGER
    lines = lp.read_text().splitlines()
    lp.write_text("\n".join(reversed(lines)) + "\n")
    assert "chain broken" in _cp_text(tmp_path, cdir)


def test_ledger_truncation_and_deletion_are_detected_only_with_the_checkpoint(tmp_path):
    _, cdir = _cp_state(tmp_path)
    lp = tmp_path / "shadow" / SH.LEDGER
    lp.write_text(lp.read_text().splitlines()[0] + "\n")                    # drop the last entry
    assert SH.check(str(tmp_path), now=NOW)["ok"]                          # without the checkpoint: invisible
    assert "run ledger truncated" in _cp_text(tmp_path, cdir)
    lp.unlink()
    assert SH.check(str(tmp_path), now=NOW)["ok"]                          # without the checkpoint: invisible
    assert "run ledger deleted" in _cp_text(tmp_path, cdir)
    assert SH.ledger_append(str(tmp_path), now=NOW, checkpoint_dir=cdir) == []   # never silently rebuilt
    assert SH.update_checkpoint(str(tmp_path), cdir, now=NOW)                      # and never absorbed


def test_journal_edit_or_truncation_is_detected_with_the_checkpoint(tmp_path):
    _, cdir = _cp_state(tmp_path)
    jp = tmp_path / "shadow" / "task_journal.jsonl"
    jp.write_text(jp.read_text().replace('"exit_code": 0', '"exit_code": 0 '))
    assert "task_journal.jsonl edited" in _cp_text(tmp_path, cdir)
    jp.write_text("")
    assert "task_journal.jsonl truncated" in _cp_text(tmp_path, cdir)


def test_appending_after_the_checkpoint_is_normal(tmp_path):
    _, cdir = _cp_state(tmp_path)
    with open(tmp_path / "shadow" / "task_journal.jsonl", "a") as f:
        f.write(json.dumps(_days("2026-11-05")[0]) + "\n")
    third = tmp_path / "shadow" / "runs" / "20261105T003000Z"
    third.mkdir(parents=True)
    (third / "run.json").write_text(json.dumps({"status": "waiting"}))
    assert SH.ledger_append(str(tmp_path), now=NOW, checkpoint_dir=cdir)
    assert SH.check(str(tmp_path), now=NOW, checkpoint_dir=cdir)["ok"]
    assert SH.update_checkpoint(str(tmp_path), cdir, now=NOW) == []


def test_consistent_replacement_of_records_ledger_and_checkpoint_is_not_detectable(tmp_path):
    """Documents the limit: with no secret and no copy off the machine, a full consistent rewrite passes."""
    run, cdir = _cp_state(tmp_path)
    (run / "data" / "BTCUSDT_1d.csv").write_bytes(b"forged")
    (tmp_path / "shadow" / SH.LEDGER).unlink()
    shutil.rmtree(cdir)
    SH.ledger_append(str(tmp_path), now=NOW, checkpoint_dir=cdir)
    SH.update_checkpoint(str(tmp_path), cdir, now=NOW)
    findings = SH.check(str(tmp_path), now=NOW, checkpoint_dir=cdir)["findings"]
    assert not any("ledger" in f or "checkpoint" in f for f in findings)
    assert any("does not match its recorded hash" in f for f in findings)   # only because run.json kept the old hash


def test_default_checkpoint_dir_is_outside_the_project(monkeypatch):
    monkeypatch.delenv("OF2_CHECKPOINT_DIR", raising=False)
    monkeypatch.delenv("OF2_STATE_DIR", raising=False)
    d = os.path.abspath(SH.default_checkpoint_dir())
    assert not d.startswith(os.path.abspath(ROOT))


# ------------------------------------------------------------------ the real Windows notifier (opt-in)
@pytest.mark.skipif(os.name != "nt" or os.environ.get("OF2_TEST_REAL_NOTIFY") != "1",
                    reason="shows a real notification; set OF2_TEST_REAL_NOTIFY=1 to run")
def test_simulated_critical_failure_produces_a_visible_windows_alert(sd):
    import shadow_notify
    hc = lambda root, now, entry: SH.check(root, now=now, pending_entry=entry)

    def simulated_failure(argv):
        d = os.path.join(sd, "runs", datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
        os.makedirs(d)
        with open(os.path.join(d, "run.json"), "w") as f:
            json.dump({"status": "SIMULATED FAILURE (alert test only; nothing is wrong)"}, f)
        return 5
    code = ST.run(tracker_main=simulated_failure, health_check=hc, notifier=shadow_notify.notify)
    assert code == 5
    note = _alerts(sd)[-1]["notification"]
    assert note["method"] == "toast" and note["delivered"], note
