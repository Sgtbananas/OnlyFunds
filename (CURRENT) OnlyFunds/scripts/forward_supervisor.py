"""Windows-compatible supervisor: keeps a forward PAPER session running and restarts it after a crash.

    python scripts/forward_supervisor.py --deployment binance-1h -- --interval 1h --every 60 --websocket

- Runs run_forward_session.py as a child process. Exit code 0 (graceful stop) ends supervision.
- Any other exit (crash, kill) -> restart after a bounded exponential back-off (10 s .. 15 min);
  the back-off resets after a child has run for 30 minutes.
- A STOP file in the deployment directory stops both supervisor and session gracefully.
- Every supervisor event is appended (UTC) to <deployment>/supervisor.log.
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STATE_ROOT = os.environ.get("OF2_STATE_DIR") or os.path.join(APP_ROOT, "state")


def log(dep, msg):
    line = f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    os.makedirs(dep, exist_ok=True)
    with open(os.path.join(dep, "supervisor.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    child_args = argv[argv.index("--") + 1:] if "--" in argv else []
    own = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser()
    ap.add_argument("--deployment", default="binance-1h")
    ap.add_argument("--min-backoff", type=float, default=10.0)
    ap.add_argument("--max-backoff", type=float, default=900.0)
    ap.add_argument("--max-restarts", type=int, default=0, help="0 = unlimited")
    ap.add_argument("--clear-stop", action="store_true", help="remove a leftover STOP file before starting")
    ap.add_argument("--request-stop", action="store_true", help="create the STOP file (graceful stop) and exit")
    a = ap.parse_args(own)
    dep = os.path.join(STATE_ROOT, "forward_deployments", a.deployment)
    stop = os.path.join(dep, "STOP")
    if a.request_stop:
        os.makedirs(dep, exist_ok=True)
        with open(stop, "w", encoding="utf-8") as f:
            f.write(datetime.now(timezone.utc).isoformat())
        log(dep, "stop requested (STOP file created)")
        return 0
    if a.clear_stop and os.path.exists(stop):
        os.remove(stop)
        log(dep, "leftover STOP file removed at start")
    cmd = [sys.executable, os.path.join(APP_ROOT, "run_forward_session.py"), "--deployment", a.deployment] + child_args
    backoff, restarts = a.min_backoff, 0
    log(dep, f"supervisor start pid={os.getpid()} cmd={' '.join(cmd[1:])}")
    while True:
        if os.path.exists(os.path.join(dep, "STOP")):
            log(dep, "STOP file present; supervisor exiting")
            return 0
        t0 = time.monotonic()
        child = subprocess.Popen(cmd, cwd=APP_ROOT)
        log(dep, f"session process started pid={child.pid}")
        code = child.wait()
        ran = time.monotonic() - t0
        if code == 0:
            log(dep, f"session exited cleanly after {ran:.0f}s; supervisor exiting")
            return 0
        restarts += 1
        if ran > 1800:
            backoff = a.min_backoff
        log(dep, f"session exited with code {code} after {ran:.0f}s; restart #{restarts} in {backoff:.0f}s")
        if a.max_restarts and restarts > a.max_restarts:
            log(dep, "max restarts exceeded; supervisor exiting")
            return 1
        slept = 0.0
        while slept < backoff:
            if os.path.exists(os.path.join(dep, "STOP")):
                log(dep, "STOP file present during back-off; supervisor exiting")
                return 0
            time.sleep(1)
            slept += 1
        backoff = min(backoff * 2, a.max_backoff)


if __name__ == "__main__":
    sys.exit(main())
