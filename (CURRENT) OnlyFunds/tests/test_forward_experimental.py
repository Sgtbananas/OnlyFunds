"""Experimental forward tracker (X1/X2): separate fingerprints, read-only use of the frozen run, isolation."""
import json
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "research"))
import forward_experimental as FE  # noqa: E402
import forward_shadow as FS  # noqa: E402
import shadow_task as ST  # noqa: E402
from tests.test_forward_shadow import shadow  # noqa: E402,F401


@pytest.fixture
def exp(shadow, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(FE, "EXP_FROZEN", str(tmp_path / "EXPERIMENTAL_FROZEN.json"))
    FS.cmd_freeze()
    assert FE.cmd_freeze() == 0
    return shadow


def latest(kind="runs"):
    d = os.path.join(FE.exp_state(), "runs")
    rid = sorted(os.listdir(d))[-1]
    with open(os.path.join(d, rid, "run.json"), encoding="utf-8") as f:
        return rid, json.load(f)


def test_freeze_is_write_once_with_separate_fingerprints(exp):
    assert FE.cmd_freeze() == 2
    fz, problems = FE.load_frozen()
    assert problems == [] and fz["fingerprints"]["X1_T50_PM"] != fz["fingerprints"]["X2_T50_PM_B"]
    assert fz["frozen_T50_fingerprint"] == FS.load_frozen()[0]["fingerprints"]["T50"]


def test_update_reads_frozen_snapshot_and_comparator_matches(exp):
    assert FS.cmd_update(skip_random=True) == 0
    frozen_before = {f: open(os.path.join(FS.STATE, "runs", r, "run.json"), "rb").read()
                     for r in os.listdir(os.path.join(FS.STATE, "runs")) for f in [r]}
    assert FE.cmd_update() == 0
    _, rec = latest()
    assert rec["status"] == "ok" and rec["anomalies"] == []          # recomputed T50 book == frozen run's book
    assert set(rec["books"]) == {"T50_comparator", "X1_T50_PM", "X2_T50_PM_B"}
    for r, b in frozen_before.items():                               # frozen records untouched
        assert open(os.path.join(FS.STATE, "runs", r, "run.json"), "rb").read() == b
    assert os.path.exists(os.path.join(FE.exp_state(), "EXPERIMENTAL_LATEST.md"))


def test_tampered_snapshot_is_refused(exp):
    FS.cmd_update(skip_random=True)
    rid = sorted(os.listdir(os.path.join(FS.STATE, "runs")))[-1]
    p = os.path.join(FS.STATE, "runs", rid, "data", "ETHUSDT_1d.csv")
    with open(p, "a", encoding="utf-8") as f:
        f.write("\n")
    assert FE.cmd_update(rid) == 6


def test_changed_experimental_code_stops_only_the_experiment(exp, monkeypatch):
    FS.cmd_update(skip_random=True)
    real = FS.file_hash
    monkeypatch.setattr(FS, "file_hash", lambda rel: "0" * 64 if rel == "research/portfolio_variants.py" else real(rel))
    assert FE.cmd_update() == 3
    monkeypatch.setattr(FS, "file_hash", real)
    assert FS.load_frozen()[1] == []                                 # frozen T50/RS3 unaffected


def test_wrapper_runs_experiment_after_frozen_run_and_journals_it(exp, monkeypatch):
    monkeypatch.setattr(ST, "state_dir", lambda: FS.STATE)
    assert ST.run(tracker_main=lambda a: FS.cmd_update(skip_random=True), experimental_main=FE.main) == 0
    with open(os.path.join(FS.STATE, "task_journal.jsonl"), encoding="utf-8") as f:
        e = json.loads(f.readlines()[-1])
    assert e["status"] == "ok" and e["experimental"]["status"] == "ok"


def test_experiment_crash_cannot_fail_the_frozen_run(exp, monkeypatch):
    monkeypatch.setattr(ST, "state_dir", lambda: FS.STATE)
    def boom(argv):
        raise RuntimeError("experimental failure")
    assert ST.run(tracker_main=lambda a: FS.cmd_update(skip_random=True), experimental_main=boom) == 0
    with open(os.path.join(FS.STATE, "task_journal.jsonl"), encoding="utf-8") as f:
        e = json.loads(f.readlines()[-1])
    assert e["status"] == "ok" and e["experimental"]["status"] == "CRASHED"


def test_experimental_code_has_no_order_paths():
    src = open(os.path.join(ROOT, "research", "forward_experimental.py"), encoding="utf-8").read()
    for bad in ("place_order", "CoinExClient", "API_SECRET", "requests", "allow_trading", "LIVE_TRADING_LOCKED ="):
        assert bad not in src


def test_experiment_import_failure_cannot_fail_the_frozen_run(exp, monkeypatch):
    monkeypatch.setattr(ST, "state_dir", lambda: FS.STATE)
    monkeypatch.setitem(sys.modules, "forward_experimental", None)          # import raises ImportError
    assert ST.run(tracker_main=lambda a: FS.cmd_update(skip_random=True), experimental_main=ST._experimental_main) == 0
    with open(os.path.join(FS.STATE, "task_journal.jsonl"), encoding="utf-8") as f:
        e = json.loads(f.readlines()[-1])
    assert e["status"] == "ok" and e["experimental"]["status"] == "CRASHED"
