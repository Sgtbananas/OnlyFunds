"""Immutable validation run storage (audit fix P2).

Layout:  <root>/<plan_name>/<run_id>/{stage1,stage2}/...
- Every run gets a unique id (UTC time + random suffix); a Stage 1 rerun always creates a new run.
- Files are written once (exclusive create) and then made read-only. Nothing is overwritten.
- Replacing a run is an explicit action: a NEW run that records ``supersedes`` and a reason;
  the old run stays untouched.
- A final-test window, once consumed, is registered (one immutable file per consumption) and any
  later run whose TEST window overlaps it is refused.
"""
from __future__ import annotations

import json
import os
import stat
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_ROOT = os.path.join(APP_ROOT, "docs", "validation", "runs")
CONSUMED_DIR_NAME = "_consumed_test_windows"


class ImmutableError(RuntimeError):
    """Raised on any attempt to overwrite or reuse something that must not change."""


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]


def write_once(path: str, data: Any, binary: bool = False) -> str:
    """Create ``path`` exclusively (refuse if it exists), then make it read-only."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0))
    except FileExistsError:
        raise ImmutableError(f"refusing to overwrite existing file {path}")
    with os.fdopen(fd, "wb") as f:
        if binary:
            f.write(data)
        else:
            text = data if isinstance(data, str) else json.dumps(data, indent=1, default=str)
            f.write(text.encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())
    os.chmod(path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    return path


class RunStore:
    def __init__(self, root: str = DEFAULT_ROOT):
        self.root = root

    def plan_dir(self, plan_name: str) -> str:
        return os.path.join(self.root, plan_name)

    def run_dir(self, plan_name: str, run_id: str) -> str:
        return os.path.join(self.plan_dir(plan_name), run_id)

    def create_run(self, plan_name: str, supersedes: Optional[str] = None, reason: str = "") -> str:
        if supersedes and not reason:
            raise ImmutableError("superseding a run requires a documented reason")
        if supersedes and not os.path.isdir(self.run_dir(plan_name, supersedes)):
            raise ImmutableError(f"cannot supersede unknown run {supersedes}")
        run_id = new_run_id()
        path = self.run_dir(plan_name, run_id)
        os.makedirs(path, exist_ok=False)  # a fresh, never-before-used directory
        write_once(os.path.join(path, "run.json"), {"run_id": run_id, "plan": plan_name,
                                                    "created_utc": datetime.now(timezone.utc).isoformat(),
                                                    "supersedes": supersedes, "reason": reason})
        return run_id

    def stage_dir(self, plan_name: str, run_id: str, stage: str, create: bool = False) -> str:
        path = os.path.join(self.run_dir(plan_name, run_id), stage)
        if create:
            try:
                os.makedirs(path, exist_ok=False)
            except FileExistsError:
                raise ImmutableError(f"{stage} already exists for run {run_id}; runs are immutable "
                                     "(create a new run, optionally with --supersedes)")
        return path

    def read_json(self, plan_name: str, run_id: str, *parts: str) -> Dict[str, Any]:
        with open(os.path.join(self.run_dir(plan_name, run_id), *parts), encoding="utf-8") as f:
            return json.load(f)

    def list_runs(self, plan_name: str) -> List[str]:
        d = self.plan_dir(plan_name)
        return sorted(x for x in os.listdir(d) if not x.startswith("_")) if os.path.isdir(d) else []

    # ------------------------------------------------------------ consumed final-test windows
    def consumed_dir(self) -> str:
        return os.path.join(self.root, CONSUMED_DIR_NAME)

    def consumed_windows(self) -> List[Dict[str, Any]]:
        d = self.consumed_dir()
        if not os.path.isdir(d):
            return []
        out = []
        for name in sorted(os.listdir(d)):
            with open(os.path.join(d, name), encoding="utf-8") as f:
                out.append(json.load(f))
        return out

    def check_test_window_unused(self, start: str, end: str, symbols: List[str], interval: str) -> None:
        s, e = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        for w in self.consumed_windows():
            ws, we = pd.Timestamp(w["start"], tz="UTC"), pd.Timestamp(w["end"], tz="UTC")
            same_market = w.get("interval") == interval and set(w.get("symbols", [])) & set(symbols)
            if s < we and ws < e and same_market:
                raise ImmutableError(
                    f"TEST window {start}..{end} overlaps a consumed final-test window {w['start']}..{w['end']} "
                    f"({w['id']}); a final test period may be used only once")

    def register_consumed(self, record_id: str, start: str, end: str, symbols: List[str], interval: str,
                          note: str) -> str:
        return write_once(os.path.join(self.consumed_dir(), f"{record_id}.json"),
                          {"id": record_id, "start": start, "end": end, "symbols": sorted(symbols),
                           "interval": interval, "note": note,
                           "registered_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
