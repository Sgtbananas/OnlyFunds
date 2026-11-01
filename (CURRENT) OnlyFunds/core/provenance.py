"""Provenance manifests for validation runs (audit fix P1).

A validation run records, in machine-readable form, everything needed to know exactly what produced
its results: the protocol and plan documents, every file that implements the validation, the
configuration, the datasets (by checksum), models, the git commit and whether the tree was clean,
the exact command, and the Python environment.

``validation_fingerprint`` hashes the parts that define *what is being validated* (protocol, plan,
implementation, configuration, data). If any of them changes, the fingerprint changes and a
later stage refuses to inherit the earlier stage's approval: a new validation version is needed.

Limitation (documented, not hidden): the original v1 evaluation (2026-11-01, commits bc0033a ->
a94145b) ran before this existed. Its implementing code was committed only after its final test,
so no fingerprint can prove which code produced it. See docs/validation/README.md.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_ROOT = os.path.dirname(APP_ROOT)
SCHEMA_VERSION = 1

# Files that materially implement validation: data loading, indicators, signals, execution,
# risk, ML, tuning, statistics, eligibility, run storage and the runner itself.
IMPLEMENTATION_FILES = [
    "core/validation.py", "core/eligibility.py", "core/run_store.py", "core/provenance.py",
    "core/backtester.py", "core/positions.py", "core/risk_manager.py", "core/core_signals.py",
    "core/indicators.py", "core/ml_filter.py", "core/autotune.py", "core/history.py",
    "core/data_quality.py", "core/coinex_client.py", "utils/config.py", "scripts/validation_run.py",
]
CONFIG_FILES = ["config/config.yaml"]
PACKAGES = ["pandas", "numpy", "scikit-learn", "scipy", "joblib", "PyYAML", "requests"]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(path: str) -> str:
    """Hash of a source/text file with CRLF normalised to LF, so the fingerprint does not depend on
    the git line-ending setting of the machine. (Data files use exact bytes: sha256_file.)"""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=30,
                              check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def git_state() -> Dict[str, Any]:
    porcelain = _git("status", "--porcelain")
    dirty_files = [line[3:] for line in porcelain.splitlines() if line.strip()]
    return {"commit": _git("rev-parse", "HEAD") or None, "branch": _git("rev-parse", "--abbrev-ref", "HEAD") or None,
            "dirty": bool(dirty_files), "dirty_files": dirty_files}


def package_versions(names: Iterable[str] = PACKAGES) -> Dict[str, Optional[str]]:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover
        return {}
    out = {}
    for n in names:
        try:
            out[n] = version(n)
        except PackageNotFoundError:
            out[n] = None
    return out


def environment() -> Dict[str, Any]:
    return {"python": sys.version.split()[0], "implementation": platform.python_implementation(),
            "platform": platform.platform(), "packages": package_versions()}


def file_hashes(rel_paths: Iterable[str], root: str = APP_ROOT) -> Dict[str, Optional[str]]:
    out = {}
    for rel in rel_paths:
        p = os.path.join(root, rel)
        out[rel] = sha256_text(p) if os.path.exists(p) else None
    return out


def dataset_provenance(symbols: Iterable[str], interval: str, exchange: str = "binance",
                       root: Optional[str] = None) -> Dict[str, Any]:
    from core.history import dataset_paths
    out = {}
    for s in symbols:
        csv, man = dataset_paths(s, interval, exchange, root)
        rec: Dict[str, Any] = {"csv": os.path.relpath(csv, APP_ROOT) if root is None else csv}
        if os.path.exists(man):
            with open(man, encoding="utf-8") as f:
                m = json.load(f)
            rec.update({"manifest_sha256": sha256_file(man), "csv_sha256_manifest": m.get("sha256"),
                        "rows": m.get("rows"), "first": m.get("quality", {}).get("first_candle"),
                        "last": m.get("quality", {}).get("last_candle")})
        rec["csv_sha256_actual"] = sha256_file(csv) if os.path.exists(csv) else None
        rec["matches_manifest"] = rec.get("csv_sha256_actual") is not None and \
            rec.get("csv_sha256_actual") == rec.get("csv_sha256_manifest")
        out[s] = rec
    return out


def fingerprint_components(protocol_path: str, plan: Dict[str, Any], data: Dict[str, Any],
                           effective_config: Dict[str, Any], impl_root: str = APP_ROOT) -> Dict[str, Any]:
    return {
        "protocol_sha256": sha256_text(protocol_path),
        "plan_sha256": sha256_json(plan),
        "implementation": file_hashes(IMPLEMENTATION_FILES, impl_root),
        "config_files": file_hashes(CONFIG_FILES, impl_root),
        "effective_config_sha256": sha256_json(effective_config),
        "data": {s: d.get("csv_sha256_actual") for s, d in data.items()},
    }


def validation_fingerprint(components: Dict[str, Any]) -> str:
    return sha256_json(components)


def diff_components(old: Dict[str, Any], new: Dict[str, Any], prefix: str = "") -> List[str]:
    """Human-readable list of what changed between two fingerprint component sets."""
    changes = []
    for k in sorted(set(old) | set(new)):
        a, b = old.get(k), new.get(k)
        if isinstance(a, dict) and isinstance(b, dict):
            changes += diff_components(a, b, f"{prefix}{k}.")
        elif a != b:
            changes.append(f"{prefix}{k}")
    return changes


def build_manifest(run_id: str, stage: str, protocol_path: str, plan: Dict[str, Any], data: Dict[str, Any],
                   effective_config: Dict[str, Any], strategies: Dict[str, Any],
                   models: Optional[Dict[str, str]] = None, extra: Optional[Dict[str, Any]] = None,
                   argv: Optional[List[str]] = None) -> Dict[str, Any]:
    comps = fingerprint_components(protocol_path, plan, data, effective_config)
    return {
        "schema": SCHEMA_VERSION, "run_id": run_id, "stage": stage,
        "command": " ".join(argv if argv is not None else sys.argv),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(), "environment": environment(),
        "protocol": {"path": os.path.relpath(protocol_path, APP_ROOT), "sha256": comps["protocol_sha256"]},
        "plan": plan, "effective_config": effective_config, "data": data, "strategies": strategies,
        "models": models or {}, "fingerprint_components": comps,
        "validation_fingerprint": validation_fingerprint(comps), **(extra or {}),
    }
