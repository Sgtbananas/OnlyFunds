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
- Provenance: the journal entry records this wrapper's SHA-256 and the Python / numpy / pandas versions, so an
  unexpected environment change is visible (the health check flags it).
- Health and alerts: after every run, still holding the lock, new runs are appended to the hash-chained run ledger
  and shadow_health.check runs. Every finding is an issue with a stable key and a level (critical / warning).
  - state/shadow/HEALTH_LATEST.json / .md: the full latest report (headline CRITICAL / WARNING / OK).
  - state/shadow/alert_state.json: the issues currently open, with first-seen and notified times.
  - state/shadow/alerts.jsonl: append-only record of every issue raised, escalated or resolved, with the result
    of the notification.
  - Notification (shadow_notify): one Windows toast per run in which an issue is new or escalated from warning to
    critical; an issue that stays open is not notified again. Critical toasts stay on screen until dismissed.
  - Exit code 11 while any critical issue is open (the tracker's own exit code wins if it failed), so Task
    Scheduler's "Last Run Result" also shows it.
  - Afterwards the checkpoint outside the state folder (shadow_health.default_checkpoint_dir) is refreshed.
  A health-check or notification failure never changes what the tracker computed or recorded.

    python research/shadow_task.py            # same as the scheduled task
"""
import hashlib
import json
import os
import platform
import sys
import traceback
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CREDENTIAL_ENV_PREFIXES = ("COINEX_", "BINANCE_API", "BINANCE_SECRET", "OKX_API", "API_KEY", "API_SECRET")
EXIT_LOCKED = 9
EXIT_CRASHED = 10
EXIT_HEALTH_ALERT = 11


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


def wrapper_sha256():
    with open(os.path.abspath(__file__), "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def environment():
    out = {"python": platform.python_version()}
    try:
        from importlib import metadata
        for pkg in ("numpy", "pandas"):
            try:
                out[pkg] = metadata.version(pkg)
            except Exception:
                out[pkg] = None
    except Exception:
        pass
    return out


def _shadow_health():
    sys.path.insert(0, HERE)
    import shadow_health
    return shadow_health


def _default_health(state_root, now, pending_entry):
    SH = _shadow_health()
    cdir = SH.default_checkpoint_dir()
    SH.ledger_append(state_root, now=now, checkpoint_dir=cdir)
    return SH.check(state_root, now=now, pending_entry=pending_entry, checkpoint_dir=cdir)


def _default_notifier(title, line1, body, level, tag):
    sys.path.insert(0, HERE)
    import shadow_notify
    return shadow_notify.notify(title, line1, body, level, tag)


def _default_checkpoint(state_root):
    SH = _shadow_health()
    return SH.update_checkpoint(state_root, SH.default_checkpoint_dir())


def _read_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


def _message(new, open_issues, now, sd):
    crit = [i for i in new if i["level"] == "critical"]
    level = "critical" if crit else "warning"
    n_crit = sum(v["level"] == "critical" for v in open_issues.values())
    title = f"OF2 shadow {level.upper()}: {len(new)} new issue{'s' if len(new) > 1 else ''}"
    local = now.astimezone()
    line1 = f"{now:%Y-%m-%d %H:%M} UTC ({local:%H:%M} local); open: {n_crit} critical, {len(open_issues) - n_crit} warning"
    shown = sorted(new, key=lambda i: i["level"] != "critical")[:3]
    body = "\n".join(f"[{i['level'].upper()}] {i['message'][:150]}" for i in shown)
    if len(new) > 3:
        body += f"\n+{len(new) - 3} more"
    body += "\nDetails: " + os.path.join(os.path.abspath(sd), "HEALTH_LATEST.md")
    tag = "of2-" + hashlib.sha256("|".join(sorted(i["key"] for i in new)).encode()).hexdigest()[:16]
    return level, title, line1, body, tag


def health_step(sd, entry, health_check, notifier=None):
    """Run the health check in its own error boundary, update the open-issue state, record and notify new issues.
    Returns (critical_open, summary)."""
    now = datetime.now(timezone.utc)
    try:
        rep = health_check(os.path.dirname(sd), now, entry)
    except BaseException:
        msg = "health check crashed: " + traceback.format_exc(limit=3).strip().splitlines()[-1]
        rep = {"ok": False, "checked_utc": now.isoformat(), "findings": [msg],
               "issues": [{"key": "health_check_crashed", "level": "critical", "message": msg}]}
    issues = rep.get("issues")
    if issues is None:                                   # a plain list of findings: treat every one as critical
        issues = [{"key": f, "level": "critical", "message": f} for f in rep.get("findings") or []]
    current = {i["key"]: i for i in issues}
    state_path = os.path.join(sd, "alert_state.json")
    open_issues = _read_json(state_path, {}).get("open", {})
    new = [i for k, i in current.items()
           if k not in open_issues or (open_issues[k]["level"] == "warning" and i["level"] == "critical")]
    resolved = [dict(open_issues[k], key=k) for k in open_issues if k not in current]
    stamp = now.isoformat(timespec="seconds")
    still = {}
    for k, i in current.items():
        o = open_issues.get(k, {})
        still[k] = {"level": i["level"], "message": i["message"], "first_seen_utc": o.get("first_seen_utc", stamp),
                    "last_seen_utc": stamp, "notified_utc": o.get("notified_utc")}
    note = None
    if new and notifier is not None:
        level, title, line1, body, tag = _message(new, still, now, sd)
        try:
            note = notifier(title, line1, body, level, tag)
        except BaseException as exc:
            note = {"method": None, "delivered": False, "detail": f"notifier failed: {type(exc).__name__}: {exc}"[:300]}
        for i in new:
            still[i["key"]]["notified_utc"] = stamp
    crit_open = any(v["level"] == "critical" for v in still.values())
    summary = {"ok": bool(rep.get("ok")), "critical_open": sum(v["level"] == "critical" for v in still.values()),
               "warnings_open": sum(v["level"] == "warning" for v in still.values()),
               "new": [i["key"] for i in new], "resolved": [r["key"] for r in resolved],
               "notification": note}
    try:
        _write_json(os.path.join(sd, "HEALTH_LATEST.json"), rep)
        head = "CRITICAL" if crit_open else ("WARNING" if still else "OK")
        lines = [f"# OF2 shadow health: {head}\n",
                 f"Checked {rep.get('checked_utc')}; run started {entry.get('started_utc')}; tracker exit "
                 f"{entry.get('exit_code')}; status: {entry.get('status')}\n",
                 "## Open issues\n"] + [f"- [{v['level'].upper()}] {v['message']} (since {v['first_seen_utc']})"
                                         for v in sorted(still.values(), key=lambda v: v["level"] != "critical")] \
            + (["- none"] if not still else [])
        lines += ["\n## Resolved in this run\n"] + [f"- {r['message']}" for r in resolved or [{"message": "none"}]]
        lines += ["\n## Info\n"] + [f"- {x}" for x in rep.get("info") or ["none"]]
        with open(os.path.join(sd, "HEALTH_LATEST.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        with open(os.path.join(sd, "alerts.jsonl"), "a", encoding="utf-8") as f:
            for i in new:
                f.write(json.dumps({"event": "raised" if i["key"] not in open_issues else "escalated", "utc": stamp,
                                    "level": i["level"], "key": i["key"], "message": i["message"],
                                    "run_started_utc": entry.get("started_utc"), "tracker_exit": entry.get("exit_code"),
                                    "notification": note}, default=str) + "\n")
            for r in resolved:
                f.write(json.dumps({"event": "resolved", "utc": stamp, "level": r["level"], "key": r["key"],
                                    "message": r["message"], "first_seen_utc": r.get("first_seen_utc"),
                                    "run_started_utc": entry.get("started_utc")}, default=str) + "\n")
        _write_json(state_path, {"updated_utc": stamp, "open": still})
    except Exception:
        summary["write_error"] = traceback.format_exc(limit=2).strip().splitlines()[-1]
    return crit_open, summary


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


def run(argv=None, tracker_main=None, experimental_main=None, health_check=None, notifier=None, checkpoint=None):
    """`health_check(state_root, now, pending_entry) -> report`, `notifier(title, line1, body, level, tag)` and
    `checkpoint(state_root)` default to shadow_health / shadow_notify in production only (tracker_main=None)."""
    sd = state_dir()
    started = datetime.now(timezone.utc)
    entry = {"started_utc": started.isoformat(), "pid": os.getpid(), "argv": argv or ["update"]}
    lock = InstanceLock(os.path.join(sd, "tracker.lock"))
    if not lock.acquire():
        entry.update(exit_code=EXIT_LOCKED, status="REFUSED: another tracker run holds the lock")
        _journal(sd, entry)
        return EXIT_LOCKED
    try:
        return _locked_run(sd, started, entry, argv, tracker_main, experimental_main, health_check, notifier,
                           checkpoint)
    except BaseException:                       # a wrapper failure is journaled too, never silent
        entry.update(exit_code=EXIT_CRASHED, status="CRASHED: wrapper error (see traceback)",
                     traceback=traceback.format_exc())
        try:
            _journal(sd, entry)
        except Exception:
            pass
        return EXIT_CRASHED
    finally:
        lock.release()


def _locked_run(sd, started, entry, argv, tracker_main, experimental_main, health_check, notifier=None,
                checkpoint=None):
    try:
        entry["wrapper_sha256"] = wrapper_sha256()
        entry["env"] = environment()
    except Exception:
        pass
    log_dir = os.path.join(sd, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"task_{started:%Y%m%d}.log")
    entry["credential_env_removed"] = strip_credentials()
    before = _runs(sd)
    old_out, old_err = sys.stdout, sys.stderr
    code = None
    production = tracker_main is None
    with open(log_path, "a", encoding="utf-8") as log:
        sys.stdout = sys.stderr = log
        print(f"=== {started.isoformat()} pid {os.getpid()} shadow update ===", flush=True)
        try:
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
    new = sorted(_runs(sd) - before)
    entry.update(exit_code=code, finished_utc=datetime.now(timezone.utc).isoformat(), log=log_path, runs=new)
    if new:
        p = os.path.join(sd, "runs", new[-1], "run.json")
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    rec = json.load(f)
            except (OSError, ValueError) as exc:
                rec = None
                entry["status"] = f"CRASHED: run.json unreadable ({type(exc).__name__})"
            if rec is not None:
                entry.update(status=rec.get("status"), evaluated_through=rec.get("evaluated_through_utc_day"),
                             anomalies=rec.get("anomalies"), problems=rec.get("problems"),
                             data_warnings=_gap_warnings(rec))
        else:
            entry["status"] = "CRASHED: run directory created but no run.json written"
    if code == EXIT_CRASHED:
        entry["status"] = "CRASHED: unhandled exception (see traceback)"
    elif "status" not in entry:
        entry["status"] = f"exit {code}: no run record produced"
    if production:
        health_check = health_check or _default_health
        notifier = notifier or _default_notifier
        checkpoint = checkpoint or _default_checkpoint
    critical_open = False
    if health_check is not None:
        critical_open, entry["health"] = health_step(sd, entry, health_check, notifier)
    _journal(sd, entry)
    if checkpoint is not None:
        try:      # refuses while the state is inconsistent; the health check has already raised that issue
            checkpoint(os.path.dirname(sd))
        except Exception:
            pass
    return EXIT_HEALTH_ALERT if (critical_open and code == 0) else code


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:] or None))
