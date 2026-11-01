"""Scheduled-task entry point for the forward shadow tracker (research only; no orders, no keys).

Task Scheduler runs this file under pythonw, which has no console. The wrapper adds what an unattended run needs.
It does not change what is evaluated: `forward_shadow.update` and its frozen fingerprints are used unchanged, and
this file is deliberately not part of the frozen implementation.

- Single instance: an exclusive OS lock on state/shadow/tracker.lock. A second run (scheduled or manual) exits
  with code 9 and logs the refusal, so tracker runs can never overlap.
- Persistent log: stdout and stderr go to state/shadow/logs/task_<UTC date>.log.
- Run journal: every invocation appends one line to state/shadow/task_journal.jsonl, with the start and end times,
  the exit code, the status of the run.json it produced, data-gap warnings and any traceback. A crash is therefore
  never silent: the journal records CRASHED and the traceback even if no run.json was written.
- No credentials: exchange-credential environment variables are removed before the tracker is imported.

    python research/shadow_task.py            # same as the scheduled task
"""
import json
import os
import sys
import traceback
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CREDENTIAL_ENV_PREFIXES = ("COINEX_", "BINANCE_API", "BINANCE_SECRET", "OKX_API", "API_KEY", "API_SECRET")
EXIT_LOCKED = 9
EXIT_CRASHED = 10


def state_dir():
    return os.path.join(os.environ.get("OF2_STATE_DIR") or os.path.join(ROOT, "state"), "shadow")


def strip_credentials(env=os.environ):
    removed = [k for k in list(env) if k.upper().startswith(CREDENTIAL_ENV_PREFIXES)]
    for k in removed:
        env.pop(k, None)
    return removed


class InstanceLock:
    """Exclusive non-blocking OS lock; released automatically if the process dies."""

    def __init__(self, path):
        self.path, self.fh = path, None

    def acquire(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.fh = open(self.path, "a+")
        try:
            if os.name == "nt":
                import msvcrt
                self.fh.seek(0)
                msvcrt.locking(self.fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.fh.close()
            self.fh = None
            return False
        return True

    def release(self):
        if self.fh is None:
            return
        try:
            if os.name == "nt":
                import msvcrt
                self.fh.seek(0)
                msvcrt.locking(self.fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
        finally:
            self.fh.close()
            self.fh = None


def _runs(sd):
    d = os.path.join(sd, "runs")
    return set(os.listdir(d)) if os.path.isdir(d) else set()


def _journal(sd, entry):
    os.makedirs(sd, exist_ok=True)
    with open(os.path.join(sd, "task_journal.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _gap_warnings(rec):
    out = []
    for s, v in (rec.get("data_status") or {}).items():
        if isinstance(v, dict):
            if v.get("missing_days"):
                out.append(f"{s}: {v['missing_days']} missing day(s) inside the snapshot")
            if v.get("stale"):
                out.append(f"{s}: stale (last complete day {v.get('last_complete_day')})")
            if v.get("note"):
                out.append(f"{s}: {v['note']}")
        else:
            out.append(f"{s}: {v}")
    return out


def _experimental_main(argv):
    import forward_experimental
    return forward_experimental.main(argv)


def run_experimental(sd, new_runs, experimental_main):
    """Experimental X1/X2 tracking (ROUND2_PROGRAMME.md): only after a successful frozen run, in its own boundary."""
    out = {"ran": False}
    if not new_runs:
        return {**out, "status": "skipped: no new frozen run"}
    p = os.path.join(sd, "runs", new_runs[-1], "run.json")
    if not os.path.exists(p):
        return {**out, "status": "skipped: frozen run produced no record"}
    with open(p, encoding="utf-8") as f:
        st = json.load(f).get("status")
    if st != "ok":
        return {**out, "status": f"skipped: frozen run status {st!r}"}
    try:
        code = experimental_main(["update", new_runs[-1]])
        return {"ran": True, "exit_code": code, "status": "ok" if code == 0 else f"exit {code}"}
    except BaseException:
        return {"ran": True, "exit_code": EXIT_CRASHED, "status": "CRASHED", "traceback": traceback.format_exc()}


def run(argv=None, tracker_main=None, experimental_main=None):
    sd = state_dir()
    started = datetime.now(timezone.utc)
    entry = {"started_utc": started.isoformat(), "pid": os.getpid(), "argv": argv or ["update"]}
    lock = InstanceLock(os.path.join(sd, "tracker.lock"))
    if not lock.acquire():
        entry.update(exit_code=EXIT_LOCKED, status="REFUSED: another tracker run holds the lock")
        _journal(sd, entry)
        return EXIT_LOCKED
    log_dir = os.path.join(sd, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"task_{started:%Y%m%d}.log")
    entry["credential_env_removed"] = strip_credentials()
    before = _runs(sd)
    old_out, old_err = sys.stdout, sys.stderr
    code = None
    with open(log_path, "a", encoding="utf-8") as log:
        sys.stdout = sys.stderr = log
        print(f"=== {started.isoformat()} pid {os.getpid()} shadow update ===", flush=True)
        try:
            production = tracker_main is None
            if production:
                sys.path.insert(0, HERE)
                import forward_shadow
                tracker_main = forward_shadow.main
            code = tracker_main(argv or ["update"])
            code = 0 if code is None else int(code)
            if production and experimental_main is None and os.path.exists(
                    os.path.join(HERE, "shadow", "EXPERIMENTAL_FROZEN.json")):
                experimental_main = _experimental_main       # imported inside its own error boundary
            if experimental_main is not None and code == 0:
                entry["experimental"] = run_experimental(sd, sorted(_runs(sd) - before), experimental_main)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
        except BaseException:
            tb = traceback.format_exc()
            print(tb, flush=True)
            entry["traceback"] = tb
            code = EXIT_CRASHED
        finally:
            sys.stdout, sys.stderr = old_out, old_err
            lock.release()
    new = sorted(_runs(sd) - before)
    entry.update(exit_code=code, finished_utc=datetime.now(timezone.utc).isoformat(), log=log_path, runs=new)
    if new:
        p = os.path.join(sd, "runs", new[-1], "run.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                rec = json.load(f)
            entry.update(status=rec.get("status"), evaluated_through=rec.get("evaluated_through_utc_day"),
                         anomalies=rec.get("anomalies"), problems=rec.get("problems"),
                         data_warnings=_gap_warnings(rec))
        else:
            entry["status"] = "CRASHED: run directory created but no run.json written"
    if code == EXIT_CRASHED:
        entry["status"] = "CRASHED: unhandled exception (see traceback)"
    elif "status" not in entry:
        entry["status"] = f"exit {code}: no run record produced"
    _journal(sd, entry)
    return code


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:] or None))
