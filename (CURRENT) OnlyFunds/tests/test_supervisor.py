"""Supervisor lifecycle: crash restart, graceful exits, single instance, headless logging (no network)."""
import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import forward_supervisor as fs  # noqa: E402

PY = sys.executable


def lines(dep):
    with open(os.path.join(dep, "supervisor.log"), encoding="utf-8") as f:
        return f.read().splitlines()


def test_crashing_session_is_restarted_with_backoff(tmp_path):
    dep = str(tmp_path)
    code = fs.supervise(dep, [PY, "-c", "import sys; print('boom'); sys.exit(7)"],
                        min_backoff=0.02, max_backoff=0.05, max_restarts=3, poll=0.01)
    log = lines(dep)
    assert code == 1
    assert sum("session process started" in x for x in log) == 4          # first run + 3 restarts
    assert any("exited with code 7" in x and "restart #1" in x for x in log)
    assert "max restarts exceeded" in log[-1]
    with open(tmp_path / "session_console.log", encoding="utf-8") as f:
        assert f.read().count("boom") == 4                                # child output captured


def test_restart_after_crash_then_clean_exit(tmp_path):
    marker = tmp_path / "ran_once"
    script = f"import os,sys\np=r'{marker}'\nif os.path.exists(p): sys.exit(0)\nopen(p,'w').close(); sys.exit(1)\n"
    code = fs.supervise(str(tmp_path), [PY, "-c", script], min_backoff=0.01, poll=0.01)
    log = lines(str(tmp_path))
    assert code == 0 and any("restart #1" in x for x in log) and "exited cleanly" in log[-1]


def test_stop_file_prevents_start_and_ends_backoff(tmp_path):
    (tmp_path / "STOP").write_text("x")
    assert fs.supervise(str(tmp_path), [PY, "-c", "raise SystemExit(1)"], poll=0.01) == 0
    assert "STOP file present" in lines(str(tmp_path))[-1]
    os.remove(tmp_path / "STOP")
    script = f"open(r'{tmp_path / 'STOP'}','w').close(); raise SystemExit(1)"   # crash, then STOP during back-off
    assert fs.supervise(str(tmp_path), [PY, "-c", script], min_backoff=30, poll=0.01) == 0
    assert "during back-off" in lines(str(tmp_path))[-1]


def test_single_instance_guard(tmp_path):
    dep = str(tmp_path)
    rival = subprocess.Popen([PY, "-c", "import time; time.sleep(30)"])
    try:
        (tmp_path / "supervisor.pid").write_text(json.dumps({"pid": rival.pid}))
        assert fs.acquire_instance(dep) == rival.pid                       # live rival -> refused
    finally:
        rival.kill()
        rival.wait()
    assert fs.acquire_instance(dep) is None                               # dead holder -> taken over
    with open(tmp_path / "supervisor.pid", encoding="utf-8") as f:
        assert json.load(f)["pid"] == os.getpid()
    fs.release_instance(dep)
    assert not (tmp_path / "supervisor.pid").exists()


def test_main_refuses_second_supervisor_and_keeps_records(tmp_path, monkeypatch):
    dep = tmp_path / "forward_deployments" / "d"
    dep.mkdir(parents=True)
    (dep / "keep.json").write_text("{}")
    rival = subprocess.Popen([PY, "-c", "import time; time.sleep(30)"])
    try:
        (dep / "supervisor.pid").write_text(json.dumps({"pid": rival.pid}))
        assert fs.main(["--deployment", "d", "--state-dir", str(tmp_path), "--clear-stop"]) == 3
    finally:
        rival.kill()
        rival.wait()
    assert (dep / "keep.json").exists() and "already running" in lines(str(dep))[-1]


def test_headless_logging_without_console(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "stdout", None)                              # as under pythonw.exe
    fs.log(str(tmp_path), "hello")
    assert lines(str(tmp_path))[-1].endswith("hello")


def test_child_uses_console_python_when_supervisor_is_pythonw(monkeypatch, tmp_path):
    (tmp_path / "pythonw.exe").write_text("")
    (tmp_path / "python.exe").write_text("")
    monkeypatch.setattr(sys, "executable", str(tmp_path / "pythonw.exe"))
    assert fs.child_python() == str(tmp_path / "python.exe")


@pytest.mark.skipif(os.name != "nt", reason="Windows launcher")
def test_launchers_use_task_scheduler_not_shell_children():
    app = os.path.join(os.path.dirname(__file__), "..")
    with open(os.path.join(app, "START_FORWARD_PAPER.bat"), encoding="utf-8") as f:
        start = f.read()
    assert "forward_task.ps1" in start and "-Action start" in start and "start \"" not in start.lower()
    with open(os.path.join(app, "scripts", "forward_task.ps1"), encoding="utf-8") as f:
        ps = f.read()
    for need in ("Register-ScheduledTask", "ExecutionTimeLimit ([TimeSpan]::Zero)", "MultipleInstances IgnoreNew",
                 "DontStopIfGoingOnBatteries", "pythonw.exe", "--request-stop"):
        assert need in ps
    assert "allow_trading" not in ps and "API_KEY" not in ps.upper().replace("NO API KEYS", "")
