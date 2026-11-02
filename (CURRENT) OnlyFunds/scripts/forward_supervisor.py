"""Windows-compatible supervisor: keeps a forward PAPER session running and restarts it after a crash.

    python scripts/forward_supervisor.py --deployment binance-1h -- --interval 1h --every 60 --websocket

For unattended runs it is launched by the Windows Task Scheduler (scripts/forward_task.ps1, used by
START_FORWARD_PAPER.bat), so it is a child of the Task Scheduler service, not of any terminal, editor or
Claude Code shell, and survives those closing.

- Runs run_forward_session.py as a child process. Exit code 0 (graceful stop) ends supervision.
- Any other exit (crash, kill) -> restart after a bounded exponential back-off (10 s .. 15 min);
  the back-off resets after a child has run for 30 minutes.
- A STOP file in the deployment directory stops both supervisor and session gracefully.
- Only one supervisor per deployment: <deployment>/supervisor.pid guards against a second one.
- Every supervisor event is appended (UTC) to <deployment>/supervisor.log; the session's own console
  output (including any traceback) is appended to <deployment>/session_console.log.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CREATE_NO_WINDOW = 0x08000000


def state_root():
    return os.environ.get("OF2_STATE_DIR") or os.path.join(APP_ROOT, "state")


def log(dep, msg):
    line = f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {msg}"
    if sys.stdout is not None:          # pythonw (Task Scheduler) has no console
        try:
            print(line, flush=True)
        except Exception:
            pass
    os.makedirs(dep, exist_ok=True)
    with open(os.path.join(dep, "supervisor.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _process_is_python(pid):
    """True if ``pid`` is a running Python process (guards against PID reuse)."""
    try:
        import psutil
        return psutil.pid_exists(pid) and "python" in psutil.Process(pid).name().lower()
    except ImportError:
        sys.path.insert(0, APP_ROOT)
        from core.paper_trader import pid_alive
        return pid_alive(pid)
    except Exception:
        return False


def acquire_instance(dep):
    """Claim the deployment for this supervisor. Returns the PID of a live rival, or None on success."""
    path = os.path.join(dep, "supervisor.pid")
    try:
        with open(path, encoding="utf-8") as f:
            other = int(json.load(f)["pid"])
        if other != os.getpid() and _process_is_python(other):
            return other
    except (FileNotFoundError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        pass
    os.makedirs(dep, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "executable": sys.executable}, f)
    os.replace(tmp, path)
    return None


def release_instance(dep):
    path = os.path.join(dep, "supervisor.pid")
    try:
        with open(path, encoding="utf-8") as f:
            mine = int(json.load(f)["pid"]) == os.getpid()
        if mine:                        # remove only after closing (Windows sharing rules)
            os.remove(path)
    except Exception:
        pass


def child_python():
    """Console python for the child (pythonw has no stdout); same installation as the supervisor."""
    exe = sys.executable
    if os.path.basename(exe).lower() == "pythonw.exe":
        cand = os.path.join(os.path.dirname(exe), "python.exe")
        if os.path.exists(cand):
            return cand
    return exe


def supervise(dep, cmd, min_backoff=10.0, max_backoff=900.0, max_restarts=0, poll=1.0):
    stop = os.path.join(dep, "STOP")
    backoff, restarts = min_backoff, 0
    flags = CREATE_NO_WINDOW if os.name == "nt" else 0
    while True:
        if os.path.exists(stop):
            log(dep, "STOP file present; supervisor exiting")
            return 0
        t0 = time.monotonic()
        with open(os.path.join(dep, "session_console.log"), "a", encoding="utf-8") as out:
            out.write(f"\n===== {datetime.now(timezone.utc).isoformat(timespec='seconds')} session start =====\n")
            out.flush()
            child = subprocess.Popen(cmd, cwd=APP_ROOT, stdin=subprocess.DEVNULL, stdout=out,
                                     stderr=subprocess.STDOUT, creationflags=flags)
            log(dep, f"session process started pid={child.pid}")
            code = child.wait()
        ran = time.monotonic() - t0
        if code == 0:
            log(dep, f"session exited cleanly after {ran:.0f}s; supervisor exiting")
            return 0
        restarts += 1
        if ran > 1800:
            backoff = min_backoff
        log(dep, f"session exited with code {code} after {ran:.0f}s; restart #{restarts} in {backoff:.0f}s")
        if max_restarts and restarts > max_restarts:
            log(dep, "max restarts exceeded; supervisor exiting")
            return 1
        slept = 0.0
        while slept < backoff:
            if os.path.exists(stop):
                log(dep, "STOP file present during back-off; supervisor exiting")
                return 0
            time.sleep(poll)
            slept += poll
        backoff = min(backoff * 2, max_backoff)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    child_args = argv[argv.index("--") + 1:] if "--" in argv else []
    own = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser()
    ap.add_argument("--deployment", default="binance-1h")
    ap.add_argument("--state-dir", default=None, help="state root (sets OF2_STATE_DIR for the session too)")
    ap.add_argument("--min-backoff", type=float, default=10.0)
    ap.add_argument("--max-backoff", type=float, default=900.0)
    ap.add_argument("--max-restarts", type=int, default=0, help="0 = unlimited")
    ap.add_argument("--clear-stop", action="store_true", help="remove a leftover STOP file before starting")
    ap.add_argument("--request-stop", action="store_true", help="create the STOP file (graceful stop) and exit")
    a = ap.parse_args(own)
    if a.state_dir:
        os.environ["OF2_STATE_DIR"] = os.path.abspath(a.state_dir)
    dep = os.path.join(state_root(), "forward_deployments", a.deployment)
    stop = os.path.join(dep, "STOP")
    if a.request_stop:
        os.makedirs(dep, exist_ok=True)
        with open(stop, "w", encoding="utf-8") as f:
            f.write(datetime.now(timezone.utc).isoformat())
        log(dep, "stop requested (STOP file created)")
        return 0
    rival = acquire_instance(dep)
    if rival:
        log(dep, f"another supervisor (pid={rival}) is already running this deployment; exiting")
        return 3
    try:
        if a.clear_stop and os.path.exists(stop):
            os.remove(stop)
            log(dep, "leftover STOP file removed at start")
        cmd = [child_python(), os.path.join(APP_ROOT, "run_forward_session.py"), "--deployment", a.deployment] + child_args
        log(dep, f"supervisor start pid={os.getpid()} parent={os.getppid()} exe={sys.executable} "
                 f"cmd={' '.join(cmd[1:])}")
        return supervise(dep, cmd, a.min_backoff, a.max_backoff, a.max_restarts)
    except BaseException as e:
        log(dep, f"supervisor error: {e!r}; {traceback.format_exc(limit=3)!r}")
        raise
    finally:
        release_instance(dep)


if __name__ == "__main__":
    sys.exit(main())
