"""Long-running forward paper session management (paper only; live trading stays locked).

Deployment directory (evidence accumulates across sessions):
  paper/ shadow.json observations/ feed_health/ steps/        shared evidence (append / atomic)
  sessions/<session_id>/session_start.json, session_end.json   write-once per process run
  sessions/<session_id>/heartbeat.json, session.log           process health, UTC log
  reports/{startup,daily,weekly,trial}/*.md   incidents/*.json  write-once
  STOP                                                         create to stop gracefully

A session that ends without a session_end.json (crash, power loss, kill) is detected by the next
session: it writes ``session_end_recovered.json`` into that session's folder (never editing its
other files) and an ``unexpected_termination`` incident.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import signal
import socket
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from core import forward_reports as R
from core.paper_trader import _atomic_write_json, pid_alive
from core.provenance import IMPLEMENTATION_FILES, environment, file_hashes, git_state, sha256_json
from core.run_store import ImmutableError, write_once
from core.trade_execution import LIVE_TRADING_LOCKED
from core.monitoring import read_json

logger = logging.getLogger("of2.session")
FORWARD_FILES = IMPLEMENTATION_FILES + ["core/live_data.py", "core/forward.py", "core/selection.py", "core/regime.py",
                                        "core/paper_trader.py", "core/session.py", "core/forward_reports.py",
                                        "core/trade_execution.py", "run_forward_session.py"]
DATA_SOURCE = {"exchange": "binance", "rest": "https://api.binance.com (public market data, allowlisted GET only)",
               "websocket": "wss://stream.binance.com:9443 (kline, bookTicker)", "credentials": "none",
               "note": "Binance data; does not establish CoinEx prices, liquidity or execution"}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_session_id() -> str:
    return utcnow().strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]


class UTCFormatter(logging.Formatter):
    converter = time.gmtime

    def formatTime(self, record, datefmt=None):
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z"


def configure_logging(log_path: Optional[str] = None, level=logging.INFO) -> None:
    fmt = UTCFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)
    if log_path:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)


class ForwardSession:
    def __init__(self, deployment_dir: str, symbols: List[str], interval: str, every_s: float,
                 engine_factory: Callable[[], Any], settings_dict: Dict[str, Any],
                 now_fn: Callable[[], datetime] = utcnow, sleep_fn: Callable[[float], None] = time.sleep,
                 max_backoff_s: float = 900.0, outage_incident_after: int = 3):
        assert LIVE_TRADING_LOCKED, "forward sessions are paper-only"
        self.dep, self.symbols, self.interval, self.every_s = deployment_dir, list(symbols), interval, every_s
        self.engine_factory, self.settings_dict = engine_factory, settings_dict
        self.now_fn, self.sleep_fn = now_fn, sleep_fn
        self.max_backoff_s, self.outage_incident_after = max_backoff_s, outage_incident_after
        self.session_id = new_session_id()
        self.sdir = os.path.join(deployment_dir, "sessions", self.session_id)
        self.stop_requested = False
        self.stop_reason = ""
        self.counters = {"steps": 0, "ok_steps": 0, "failed_steps": 0, "consecutive_failures": 0, "incidents": 0,
                         "exceptions": 0}
        self.engine = None
        self._in_outage = False
        self._last_day = None
        self._last_week = None
        self.last_step: Dict[str, Any] = {}

    # ---------------------------------------------------------- lifecycle
    def start(self) -> Dict[str, Any]:
        os.makedirs(os.path.join(self.dep, "sessions"), exist_ok=True)
        recovered = self.recover_previous_sessions()
        os.makedirs(self.sdir, exist_ok=False)
        configure_logging(os.path.join(self.sdir, "session.log"))
        start = {
            "session_id": self.session_id, "start_utc": self.now_fn().isoformat(timespec="seconds"),
            "pid": os.getpid(), "host": socket.gethostname(), "platform": platform.platform(),
            "git": git_state(), "environment": environment(),
            "config_fingerprint": sha256_json({"files": file_hashes(FORWARD_FILES + ["config/config.yaml"]),
                                               "settings": self.settings_dict}),
            "implementation": file_hashes(FORWARD_FILES), "settings": self.settings_dict,
            "data_source": DATA_SOURCE, "universe": self.symbols, "interval": self.interval, "step_every_s": self.every_s,
            "live_trading_locked": LIVE_TRADING_LOCKED, "recovered_previous_sessions": recovered,
        }
        write_once(os.path.join(self.sdir, "session_start.json"), start)
        self._install_signal_handlers()
        self.engine = self.engine_factory()
        logger.info("session %s started (git %s, dirty=%s)", self.session_id, start["git"]["commit"], start["git"]["dirty"])
        self.startup_report(start)
        self._last_day = self.now_fn().date()
        self._last_week = self.now_fn().isocalendar()[:2]
        return start

    def recover_previous_sessions(self) -> List[str]:
        out = []
        root = os.path.join(self.dep, "sessions")
        for sid in sorted(os.listdir(root)) if os.path.isdir(root) else []:
            d = os.path.join(root, sid)
            if not os.path.exists(os.path.join(d, "session_start.json")):
                continue
            if any(os.path.exists(os.path.join(d, f)) for f in ("session_end.json", "session_end_recovered.json")):
                continue
            start = read_json(os.path.join(d, "session_start.json"))
            if start.get("pid") and pid_alive(int(start["pid"])):   # includes our own pid: never two sessions at once
                raise RuntimeError(f"session {sid} (pid {start['pid']}) is still running; refusing a second session "
                                   "on the same deployment")
            hb_path = os.path.join(d, "heartbeat.json")
            hb = read_json(hb_path) if os.path.exists(hb_path) else {}
            rec = {"session_id": sid, "ended": "unexpectedly (no graceful shutdown record)",
                   "last_heartbeat_utc": hb.get("time"), "detected_by_session": self.session_id,
                   "detected_utc": self.now_fn().isoformat(timespec="seconds"), "counters_at_last_heartbeat": hb.get("counters")}
            write_once(os.path.join(d, "session_end_recovered.json"), rec)
            R.incident(self.dep, self.session_id, "unexpected_termination", rec)
            self.counters["incidents"] += 1
            out.append(sid)
        return out

    def _install_signal_handlers(self) -> None:
        def handler(signum, _frame):
            self.request_stop(f"signal {signum}")
        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            if hasattr(signal, name):
                try:
                    signal.signal(getattr(signal, name), handler)
                except (ValueError, OSError):   # not in main thread (tests)
                    pass

    def request_stop(self, reason: str) -> None:
        self.stop_requested, self.stop_reason = True, reason

    def stop_file(self) -> str:
        return os.path.join(self.dep, "STOP")

    # ---------------------------------------------------------- main loop
    def run(self, max_steps: Optional[int] = None, run_for_s: Optional[float] = None) -> Dict[str, Any]:
        t_end = time.monotonic() + run_for_s if run_for_s else None
        try:
            while not self.stop_requested:
                if os.path.exists(self.stop_file()):
                    self.request_stop("STOP file present")
                    break
                ok = self.one_step()
                self.periodic_reports()
                if max_steps and self.counters["steps"] >= max_steps:
                    self.request_stop("max_steps reached")
                    break
                if t_end and time.monotonic() >= t_end:
                    self.request_stop("run duration reached")
                    break
                self.sleep_fn(self.next_delay(ok))
        except KeyboardInterrupt:
            self.request_stop("KeyboardInterrupt")
        except Exception as exc:      # never vanish silently
            self.counters["exceptions"] += 1
            logger.exception("session loop crashed")
            R.incident(self.dep, self.session_id, "session_crash", {"error": f"{type(exc).__name__}: {exc}"})
            self.request_stop(f"crash: {type(exc).__name__}")
            raise
        finally:
            self.shutdown()
        return self.counters

    def next_delay(self, ok: bool) -> float:
        if ok:
            return self.every_s
        k = self.counters["consecutive_failures"]
        return min(self.every_s * (2 ** min(k, 10)), self.max_backoff_s)   # bounded exponential back-off

    def one_step(self) -> bool:
        self.counters["steps"] += 1
        now = self.now_fn()
        rec: Dict[str, Any] = {"time": now.isoformat(), "session_id": self.session_id}
        try:
            out = self.engine.step()
            errors = out.get("errors", [])
            ok = len(errors) < len(self.symbols)
            rec.update({"ok": ok, "observed": out.get("observed", []), "errors": errors,
                        "trader_actions": [d.get("action") for d in out.get("trader", [])],
                        "shadow_resolved": len(out.get("shadow_resolved", []))})
        except Exception as exc:
            self.counters["exceptions"] += 1
            logger.exception("step failed")
            ok = False
            rec.update({"ok": False, "errors": [f"{type(exc).__name__}: {exc}"]})
            if "Corrupt" in type(exc).__name__:
                R.incident(self.dep, self.session_id, "ledger_corrupt", {"error": str(exc)})
                self.request_stop("ledger corrupt")
        if ok:
            self.counters["ok_steps"] += 1
            if self._in_outage:
                R.incident(self.dep, self.session_id, "outage_recovered",
                           {"failed_steps": self.counters["consecutive_failures"]})
                self.counters["incidents"] += 1
                self._in_outage = False
            self.counters["consecutive_failures"] = 0
        else:
            self.counters["failed_steps"] += 1
            self.counters["consecutive_failures"] += 1
            if self.counters["consecutive_failures"] == self.outage_incident_after and not self._in_outage:
                self._in_outage = True
                R.incident(self.dep, self.session_id, "data_outage",
                           {"consecutive_failed_steps": self.counters["consecutive_failures"], "last_errors": rec.get("errors")})
                self.counters["incidents"] += 1
        self.last_step = rec
        self._append(os.path.join(self.dep, "steps", f"{now.strftime('%Y-%m-%d')}.jsonl"), rec)
        self.heartbeat()
        return ok

    def heartbeat(self) -> None:
        hb = {"time": self.now_fn().isoformat(), "pid": os.getpid(), "counters": self.counters,
              "feed": self.engine.live.feed_health() if self.engine is not None and hasattr(self.engine, "live") else None}
        _atomic_write_json(os.path.join(self.sdir, "heartbeat.json"), hb)

    @staticmethod
    def _append(path: str, rec: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    # ---------------------------------------------------------- reports
    def startup_report(self, start: Dict[str, Any]) -> Optional[str]:
        checks = {"live_trading_locked": LIVE_TRADING_LOCKED, "credentials_required": False,
                  "approved_strategies": len(getattr(getattr(self.engine, "selector", None), "approvals", []) or []),
                  "expected_behaviour": "observe, log and shadow-track; no positions unless a strategy is validated",
                  "disk": R.disk_status(self.dep)}
        self.one_step()                               # first step through the normal, logged path
        checks["first_step"] = self.last_step
        live = getattr(self.engine, "live", None)
        if live is not None:
            checks["feed"] = live.feed_health()
            checks["exchange_clock_offset_ms"] = getattr(getattr(live, "client", None), "time_offset_ms", None)
        led = getattr(getattr(self.engine, "trader", None), "reconcile", None)
        if led:
            checks["ledger_reconcile"] = led() or "consistent"
        text = R.render(f"OF2 forward paper — startup health report, session {self.session_id}",
                        {"Session": {k: start[k] for k in ("session_id", "start_utc", "git", "config_fingerprint",
                                                           "data_source", "universe", "interval", "recovered_previous_sessions")},
                         "Health checks": checks})
        return R.write_report(self.dep, "startup", self.session_id, text)

    def periodic_reports(self) -> None:
        now = self.now_fn()
        day = now.date()
        if self._last_day and day != self._last_day:
            s = pd.Timestamp(self._last_day, tz="UTC")
            R.write_report(self.dep, "daily", str(self._last_day), R.period_report(self.dep, "daily", s,
                                                                                  s + pd.Timedelta(days=1), self.every_s))
            self._last_day = day
        week = now.isocalendar()[:2]
        if self._last_week and week != self._last_week:
            y, w = self._last_week
            s = pd.Timestamp(datetime.fromisocalendar(y, w, 1), tz="UTC")
            R.write_report(self.dep, "weekly", f"{y}-W{w:02d}", R.period_report(self.dep, "weekly", s,
                                                                                s + pd.Timedelta(days=7), self.every_s))
            self._last_week = week

    def shutdown(self) -> None:
        end = {"session_id": self.session_id, "end_utc": self.now_fn().isoformat(timespec="seconds"),
               "reason": self.stop_reason or "unknown", "counters": self.counters}
        try:
            if self.engine is not None and hasattr(self.engine, "live"):
                end["feed_health"] = self.engine.live.feed_health()
                self.engine.live.close()
        except Exception as exc:
            end["close_error"] = str(exc)
        try:
            write_once(os.path.join(self.sdir, "session_end.json"), end)
        except ImmutableError:
            pass
        logger.info("session %s ended: %s", self.session_id, end["reason"])
        for h in list(logging.getLogger().handlers):      # release the session log file (Windows file locks)
            if isinstance(h, logging.FileHandler) and h.baseFilename.startswith(os.path.abspath(self.sdir)):
                logging.getLogger().removeHandler(h)
                h.close()
