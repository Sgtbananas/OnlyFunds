"""Append-only, hash-chained hypothesis register (pre-registration for human- and AI-proposed hypotheses).

Every hypothesis is recorded BEFORE it is tested: its statement, source (human or AI), the evaluation protocol,
the decision rule and the data it may touch. Results are appended later as separate entries that reference the
hypothesis id. Each line stores the SHA-256 of the previous line, so editing or deleting an earlier entry breaks
the chain and `verify()` reports it. Nothing is ever rewritten.

The register is analysis-only. It cannot change frozen strategies, risk limits or execution, and it holds no
market data. An AI-proposed hypothesis gets the same protocol and the same decision rules as any other.

    python research/hypotheses.py list
    python research/hypotheses.py verify
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REGISTER = os.path.join(ROOT, "research", "registry", "hypotheses.jsonl")
GENESIS = "0" * 64
SOURCES = ("human", "ai")
FORBIDDEN_DATA_FROM = "2025-01-01"      # consumed final-test window; never used for development


def _lines(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [x.rstrip("\n") for x in f if x.strip()]


def _digest(line):
    return hashlib.sha256(line.encode("utf-8")).hexdigest()


def entries(path=REGISTER):
    return [json.loads(x) for x in _lines(path)]


def verify(path=REGISTER):
    """Returns a list of problems (empty = chain intact)."""
    problems, prev = [], GENESIS
    seen = set()
    for i, line in enumerate(_lines(path)):
        e = json.loads(line)
        if e.get("prev_sha256") != prev:
            problems.append(f"line {i + 1}: chain broken (prev_sha256 mismatch)")
        if e["kind"] == "hypothesis":
            if e["id"] in seen:
                problems.append(f"line {i + 1}: duplicate hypothesis id {e['id']}")
            seen.add(e["id"])
        elif e["kind"] == "result" and e["id"] not in seen:
            problems.append(f"line {i + 1}: result for unregistered hypothesis {e['id']}")
        prev = _digest(line)
    return problems


def _append(entry, path):
    lines = _lines(path)
    entry = {**entry, "prev_sha256": _digest(lines[-1]) if lines else GENESIS}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def register(hid, statement, source, protocol, decision_rule, data_window, proposer="", path=REGISTER):
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}")
    if data_window[1] > FORBIDDEN_DATA_FROM:
        raise ValueError(f"data window {data_window} reaches into the consumed test window (>= {FORBIDDEN_DATA_FROM})")
    if any(e["kind"] == "hypothesis" and e["id"] == hid for e in entries(path)):
        raise ValueError(f"hypothesis {hid} is already registered; register a new id instead of editing")
    return _append({"kind": "hypothesis", "id": hid, "statement": statement, "source": source, "proposer": proposer,
                    "protocol": protocol, "decision_rule": decision_rule, "data_window": list(data_window),
                    "registered_utc": datetime.now(timezone.utc).isoformat()}, path)


def record_result(hid, verdict, summary, evidence_ref, path=REGISTER):
    if not any(e["kind"] == "hypothesis" and e["id"] == hid for e in entries(path)):
        raise ValueError(f"{hid} was never registered; results cannot be recorded for unregistered hypotheses")
    if verdict not in ("supported", "rejected", "inconclusive"):
        raise ValueError("verdict must be supported, rejected or inconclusive")
    return _append({"kind": "result", "id": hid, "verdict": verdict, "summary": summary,
                    "evidence_ref": evidence_ref, "recorded_utc": datetime.now(timezone.utc).isoformat()}, path)


def registered_before(hid, when_iso, path=REGISTER):
    for e in entries(path):
        if e["kind"] == "hypothesis" and e["id"] == hid:
            return e["registered_utc"] < when_iso
    return False


def main(argv=None):
    cmd = (argv or sys.argv[1:] or ["list"])[0]
    if cmd == "verify":
        p = verify()
        print("chain intact" if not p else "\n".join(p))
        return 1 if p else 0
    for e in entries():
        if e["kind"] == "hypothesis":
            print(f"{e['id']:10s} [{e['source']}] {e['registered_utc'][:19]}  {e['statement'][:90]}")
        else:
            print(f"{e['id']:10s}   -> {e['verdict']}: {e['summary'][:90]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
