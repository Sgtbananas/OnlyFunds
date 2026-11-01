"""The read-only shadow health check detects tampered snapshots, crashes and dangling experimental runs."""
import hashlib
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "research"))
import shadow_health as SH  # noqa: E402


def _state(tmp_path, journal=(), exp_ref=None):
    import forward_shadow as FS
    fz = FS.load_frozen()[0]
    run = tmp_path / "shadow" / "runs" / "20261103T003000Z"
    (run / "data").mkdir(parents=True)
    csv = b"timestamp,Open\n2026-11-02,1\n"
    (run / "data" / "BTCUSDT_1d.csv").write_bytes(csv)
    (run / "run.json").write_text(json.dumps({"status": "ok", "frozen_fingerprints": fz["fingerprints"],
                                              "data_status": {"BTCUSDT": {"sha256": hashlib.sha256(csv).hexdigest()}}}))
    with open(tmp_path / "shadow" / "task_journal.jsonl", "w") as f:
        for e in journal:
            f.write(json.dumps(e) + "\n")
    if exp_ref:
        er = tmp_path / "shadow_experimental" / "runs" / exp_ref
        er.mkdir(parents=True)
        (er / "run.json").write_text(json.dumps({"frozen_run": exp_ref}))
    return run


def test_clean_state_passes_and_snapshots_are_verified(tmp_path):
    _state(tmp_path, [{"exit_code": 0, "status": "ok"}, {"exit_code": 9, "status": "REFUSED"}], "20261103T003000Z")
    rep = SH.check(str(tmp_path))
    assert rep["ok"], rep["findings"]
    assert rep["snapshots_verified"] == 1 and rep["journal"]["refused_overlap"] == 1


def test_tampered_snapshot_and_crash_are_reported(tmp_path):
    run = _state(tmp_path, [{"exit_code": 10, "status": "CRASHED: unhandled exception"}], "20991231T000000Z")
    (run / "data" / "BTCUSDT_1d.csv").write_bytes(b"timestamp,Open\n2026-11-02,2\n")
    rep = SH.check(str(tmp_path))
    text = " | ".join(rep["findings"])
    assert not rep["ok"]
    assert "does not match its recorded hash" in text and "crash" in text and "does not exist" in text


def test_tracking_code_has_no_order_capability():
    assert not [f for f in SH.check()["findings"] if "order/credential pattern" in f]
