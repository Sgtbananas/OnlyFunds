"""Shadow monitoring: missed runs, evaluation lag, anomalies, snapshot data quality, coin drop-outs, experimental
failures, environment drift, the run ledger and the wrapper's alerting. Offline, temporary state only."""
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "research"))
import shadow_health as SH  # noqa: E402
import shadow_task as ST  # noqa: E402
from tests.test_forward_shadow import shadow  # noqa: E402,F401  (offline fixture: synthetic data, network blocked)

NOW = datetime(2026, 11, 5, 6, 0, tzinfo=timezone.utc)
CSV = (b"timestamp,Open,High,Low,Close,Volume\n2026-11-01 00:00:00+00:00,1,1.2,0.9,1.1,5\n"
       b"2026-11-02 00:00:00+00:00,1.1,1.3,1.0,1.2,6\n")


def _state(tmp_path, journal=(), rid="20261103T003000Z"):
    import forward_shadow as FS
    fz = FS.load_frozen()[0]
    run = tmp_path / "shadow" / "runs" / rid
    (run / "data").mkdir(parents=True)
    (run / "data" / "BTCUSDT_1d.csv").write_bytes(CSV)
    (run / "run.json").write_text(json.dumps({"status": "ok", "frozen_fingerprints": fz["fingerprints"],
                                              "data_status": {"BTCUSDT": {"sha256": hashlib.sha256(CSV).hexdigest()}}}))
    with open(tmp_path / "shadow" / "task_journal.jsonl", "w") as f:
        for e in journal:
            f.write(json.dumps(e) + "\n")
    return run


def _days(*days, lag=1, **extra):
    out = []
    for d in days:
        ev = (datetime.fromisoformat(d) - timedelta(days=lag)).date().isoformat()
        out.append({"started_utc": f"{d}T00:30:05+00:00", "exit_code": 0, "status": "ok", "evaluated_through": ev,
                    **extra})
    return out


def _rewrite(run, **changes):
    rec = json.loads((run / "run.json").read_text())
    rec.update(changes)
    (run / "run.json").write_text(json.dumps(rec))


def _text(tmp_path, now=NOW):
    return " | ".join(SH.check(str(tmp_path), now=now)["findings"])


# ------------------------------------------------------------------ health check
def test_daily_runs_without_gaps_pass(tmp_path):
    _state(tmp_path, _days("2026-11-03", "2026-11-04", "2026-11-05"))
    rep = SH.check(str(tmp_path), now=NOW)
    assert rep["ok"], rep["findings"]
    assert rep["evaluation_lag_days"] == 1 and rep["last_successful_run_day"] == "2026-11-05"


def test_missed_day_is_reported(tmp_path):
    _state(tmp_path, _days("2026-11-03", "2026-11-05"))
    assert "missed runs" in _text(tmp_path) and "2026-11-04" in _text(tmp_path)


def test_no_run_today_is_reported_only_after_the_grace_period(tmp_path):
    _state(tmp_path, _days("2026-11-03", "2026-11-04"))
    early = SH.check(str(tmp_path), now=datetime(2026, 11, 5, 2, 0, tzinfo=timezone.utc))
    assert early["ok"], early["findings"]
    assert "2026-11-05" in _text(tmp_path)


def test_evaluation_lag_is_reported(tmp_path):
    _state(tmp_path, _days("2026-11-05", lag=3))
    assert "evaluation lags" in _text(tmp_path)


def test_anomalies_and_abnormal_status_are_reported(tmp_path):
    run = _state(tmp_path)
    _rewrite(run, anomalies=["DATA_REVISION: T50 closed trades before 2026-11-04 differ"],
             status="FETCH_FAILED: nothing computed")
    t = _text(tmp_path)
    assert "DATA_REVISION" in t and "FETCH_FAILED" in t


def test_bad_snapshot_data_is_reported(tmp_path):
    run = _state(tmp_path)
    broken = (b"timestamp,Open,High,Low,Close,Volume\n2026-11-01 00:00:00+00:00,1,0.95,0.9,1.1,5\n"
              b"2026-11-04 00:00:00+00:00,1.1,1.3,1.0,1.2,6\n")
    (run / "data" / "BTCUSDT_1d.csv").write_bytes(broken)
    _rewrite(run, data_status={"BTCUSDT": {"sha256": hashlib.sha256(broken).hexdigest()}})
    t = _text(tmp_path)
    assert "inconsistent OHLC" in t and "2 missing days" in t


def test_snapshot_quality_catches_nan_duplicates_and_non_daily():
    rows = (b"timestamp,Open,High,Low,Close,Volume\n2026-11-01 00:00:00+00:00,1,1.2,0.9,,5\n"
            b"2026-11-01 00:00:00+00:00,1,1.2,0.9,1.1,5\n2026-11-02 12:00:00+00:00,1,1.2,0.9,1.1,-1\n")
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        f.write(rows)
    probs = " | ".join(SH.snapshot_quality(f.name))
    os.unlink(f.name)
    assert "NaN" in probs and "duplicate" in probs and "negative volume" in probs and "not at 00:00" in probs


def test_coin_dropping_out_between_runs_is_reported(tmp_path):
    run = _state(tmp_path)
    fresh = {"sha256": hashlib.sha256(CSV).hexdigest(), "stale": False}
    (run / "data" / "ETHUSDT_1d.csv").write_bytes(CSV)
    _rewrite(run, data_status={"BTCUSDT": fresh, "ETHUSDT": fresh})
    later = run.parent / "20261104T003000Z"
    (later / "data").mkdir(parents=True)
    (later / "data" / "BTCUSDT_1d.csv").write_bytes(CSV)
    rec = json.loads((run / "run.json").read_text())
    rec["data_status"] = {"BTCUSDT": fresh, "ETHUSDT": "no data returned (delisted or not trading)"}
    (later / "run.json").write_text(json.dumps(rec))
    assert "ETHUSDT was fresh" in _text(tmp_path)


def test_known_unavailable_coin_is_info_not_a_finding(tmp_path):
    run = _state(tmp_path, _days("2026-11-05"))
    _rewrite(run, data_status={"BTCUSDT": {"sha256": hashlib.sha256(CSV).hexdigest()},
                               "EOSUSDT": "no data returned (delisted or not trading)"})
    rep = SH.check(str(tmp_path), now=NOW)
    assert rep["ok"], rep["findings"]
    assert any("EOSUSDT" in i for i in rep["info"])


def test_experimental_failures_are_reported(tmp_path):
    _state(tmp_path, _days("2026-11-05", experimental={"ran": True, "exit_code": 2, "status": "exit 2"}))
    er = tmp_path / "shadow_experimental" / "runs" / "20261103T003000Z"
    er.mkdir(parents=True)
    (er / "run.json").write_text(json.dumps({"frozen_run": "20261103T003000Z", "status": "ok",
                                             "anomalies": ["COMPARATOR_MISMATCH: frozen 1 vs recomputed 2"]}))
    t = _text(tmp_path)
    assert "experimental step exit 2" in t and "COMPARATOR_MISMATCH" in t


def test_environment_change_is_flagged_only_while_it_is_the_latest_run(tmp_path):
    a = {"python": "3.9.13", "numpy": "1.26.4", "pandas": "2.2.2"}
    b = {**a, "pandas": "2.3.0"}
    _state(tmp_path, _days("2026-11-04", env=a) + _days("2026-11-05", env=b))
    assert "environment changed" in _text(tmp_path)
    with open(tmp_path / "shadow" / "task_journal.jsonl", "a") as f:
        f.write(json.dumps(_days("2026-11-06", env=b)[0]) + "\n")
    rep = SH.check(str(tmp_path), now=NOW + timedelta(days=1))
    assert not any("environment changed" in f for f in rep["findings"])
    assert rep["journal"]["environment_changes"][0]["changes"] == {"pandas": ["2.2.2", "2.3.0"]}


def test_run_ledger_detects_changed_and_deleted_runs(tmp_path):
    run = _state(tmp_path, _days("2026-11-03", "2026-11-04", "2026-11-05"))
    added = SH.ledger_append(str(tmp_path), now=NOW)
    assert [e["run_id"] for e in added] == ["20261103T003000Z"] and added[0]["backfilled"]
    assert SH.ledger_append(str(tmp_path), now=NOW) == []                       # idempotent
    rep = SH.check(str(tmp_path), now=NOW)
    assert rep["ok"], rep["findings"]
    (run / "run.json").write_text((run / "run.json").read_text() + " ")
    assert "run.json changed after it was recorded" in _text(tmp_path)
    shutil.rmtree(run)
    assert "was deleted" in _text(tmp_path)


def test_run_ledger_chain_tampering_is_detected(tmp_path):
    _state(tmp_path)
    second = tmp_path / "shadow" / "runs" / "20261104T003000Z"
    second.mkdir(parents=True)
    (second / "run.json").write_text(json.dumps({"status": "waiting"}))
    SH.ledger_append(str(tmp_path), now=NOW)
    lp = tmp_path / "shadow" / SH.LEDGER
    lines = lp.read_text().splitlines()
    first = json.loads(lines[0])
    first["files"]["run.json"] = "0" * 64                                     # forge a recorded hash
    lp.write_text("\n".join([json.dumps(first, sort_keys=True)] + lines[1:]) + "\n")
    assert "chain broken" in _text(tmp_path)


