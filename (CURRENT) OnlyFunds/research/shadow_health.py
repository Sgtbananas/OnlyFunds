"""Health check of forward shadow tracking (frozen T50/RS3 and experimental X1/X2). Research only.

`check` is read-only: it reads state/, the freeze files and (optionally) the checkpoint, and returns a report. Every
finding is an issue with a stable key and a level:
- critical: the forward record may be wrong, incomplete or altered, or tracking did not run;
- warning:  worth a look, results not known to be wrong (evaluation lag, a coin dropping out, a package upgrade).
The scheduled wrapper (shadow_task.py) runs it after every update, appends new runs to the run ledger
(`ledger_append`), refreshes the checkpoint (`update_checkpoint`), and notifies the operator once per new issue.

Checks
1. Freeze files: the frozen T50/RS3 implementation and spec fingerprints, and the experimental X1/X2 ones, match.
2. Every stored frozen run: run.json is readable, its status is a normal one ("ok" or "waiting"), it records no
   anomalies (DATA_REVISION etc.), and every snapshot CSV still matches its recorded SHA-256.
3. Run journal: every invocation has an exit code; crashes, non-zero exits, failed experimental steps and
   package-version changes are reported.
4. Missed runs: every UTC day from the first journal entry to today (after a grace period past 00:30 UTC) has a
   successful tracker run; the evaluated day does not lag the run date by more than one day.
5. Data quality of the latest snapshot: OHLC present, positive and consistent, no duplicate, out-of-order or
   missing days; a coin fresh in the previous run but now stale or no longer returned is flagged.
6. Experimental runs reference an existing frozen run, finished "ok" without anomalies, and live outside the
   frozen state directory.
7. Run ledger (state/shadow/run_ledger.jsonl): a SHA-256 hash chain over every file of every stored run, plus a
   checkpoint of the ledger head and of the append-only logs kept OUTSIDE the state folder (see "Integrity").
8. No exchange-order capability: no order, signing or credential code in the tracking files.

Integrity: what the ledger and checkpoint can and cannot detect (no secret key is used, so nothing here is
tamper-proof against someone who can write to this machine):
- a stored run file changed or deleted, or a whole run deleted: detected (ledger file hashes);
- a ledger entry edited, deleted from the middle, or entries reordered: detected (hash chain);
- the last ledger entries removed (truncation) or the whole ledger deleted: detected only through the checkpoint;
  without one, a truncated or missing ledger looks like a ledger that was never written;
- the journal or alert log edited or truncated: detected through the checkpoint (byte length + prefix hash);
- runs, ledger and checkpoint all rewritten consistently: NOT detectable locally. A copy of the ledger head kept
  off this machine (written down, or stored elsewhere by the operator) is the only remedy.

    python research/shadow_health.py
"""
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

TRACKING_FILES = ["research/forward_shadow.py", "research/forward_experimental.py", "research/shadow_task.py",
                  "research/portfolio.py", "research/portfolio_variants.py", "research/round1.py"]
ORDER_PATTERNS = [r"place_order", r"create_order", r"cancel_order", r"/api/v\d/order", r"hmac\.new", r"X-MBX-APIKEY",
                  r"CoinExClient", r"requests\.post", r"API_SECRET\s*="]
NORMAL_STATUS = ("ok", "waiting")
GRACE_AFTER_SCHEDULE = timedelta(hours=3)      # the task runs at 00:30 UTC; a day counts as missed after 03:30 UTC
SCHEDULE_UTC = timedelta(minutes=30)
LEDGER = "run_ledger.jsonl"
CHECKPOINT = "ledger_checkpoint.json"
APPEND_ONLY_LOGS = ("task_journal.jsonl", "alerts.jsonl")
CRITICAL, WARNING = "critical", "warning"


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _canon(d):
    return json.dumps(d, sort_keys=True, separators=(",", ":"), default=str).encode()


def _day(ts):
    try:
        return datetime.fromisoformat(str(ts)).astimezone(timezone.utc).date()
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------ run ledger (hash chain)
def _run_files(run_dir):
    out = {}
    for base, _, files in os.walk(run_dir):
        for fn in files:
            p = os.path.join(base, fn)
            out[os.path.relpath(p, run_dir).replace(os.sep, "/")] = _sha(p)
    return dict(sorted(out.items()))


def _ledger_entries(shadow):
    p = os.path.join(shadow, LEDGER)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def _scopes(state_root):
    return {"frozen": os.path.join(state_root, "shadow", "runs"),
            "experimental": os.path.join(state_root, "shadow_experimental", "runs")}


def ledger_append(state_root, now=None, checkpoint_dir=None):
    """Append every stored run (frozen and experimental) not yet in the ledger. Returns the entries added.

    A run first seen after the run that created it (a manual tracker run, or runs made before the ledger existed)
    is marked backfilled: its hashes prove it has not changed since then, not since its creation. If the checkpoint
    shows the ledger was truncated, deleted or rewritten, nothing is appended (that would hide the evidence)."""
    if checkpoint_dir is not None and checkpoint_problems(state_root, checkpoint_dir, logs=False):
        return []
    shadow = os.path.join(state_root, "shadow")
    entries = _ledger_entries(shadow) or []
    seen = {(e["scope"], e["run_id"]) for e in entries}
    prev = entries[-1]["hash"] if entries else "0" * 64
    now = now or datetime.now(timezone.utc)
    added = []
    for scope, d in _scopes(state_root).items():
        for rid in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if (scope, rid) in seen or not os.path.isdir(os.path.join(d, rid)):
                continue
            age = now - datetime.strptime(rid[:15], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc) \
                if re.match(r"^\d{8}T\d{6}", rid) else None
            e = {"seq": len(entries) + len(added), "scope": scope, "run_id": rid,
                 "recorded_utc": now.isoformat(timespec="seconds"),
                 "backfilled": age is None or age > timedelta(hours=3),
                 "files": _run_files(os.path.join(d, rid)), "prev": prev}
            e["hash"] = hashlib.sha256(_canon(e)).hexdigest()
            prev = e["hash"]
            added.append(e)
    if added:
        os.makedirs(shadow, exist_ok=True)
        with open(os.path.join(shadow, LEDGER), "a", encoding="utf-8") as f:
            for e in added:
                f.write(json.dumps(e, sort_keys=True) + "\n")
    return added


def verify_ledger(state_root):
    """Returns (findings, info)."""
    shadow = os.path.join(state_root, "shadow")
    entries = _ledger_entries(shadow)
    findings, info = [], []
    if not entries:
        info.append("run ledger not created yet, or deleted (only the checkpoint can tell which)")
        return findings, info
    prev = "0" * 64
    dirs = _scopes(state_root)
    for i, e in enumerate(entries):
        body = {k: v for k, v in e.items() if k != "hash"}
        if e.get("prev") != prev or hashlib.sha256(_canon(body)).hexdigest() != e.get("hash") or e.get("seq") != i:
            findings.append(f"run ledger: chain broken at entry {i} ({e.get('scope')} {e.get('run_id')}): an entry was "
                            "edited, removed or reordered")
            break
        prev = e["hash"]
        rd = os.path.join(dirs.get(e["scope"], ""), e["run_id"])
        if not os.path.isdir(rd):
            findings.append(f"run ledger: {e['scope']} run {e['run_id']} was deleted")
            continue
        for rel, h in e["files"].items():
            p = os.path.join(rd, *rel.split("/"))
            if not os.path.exists(p):
                findings.append(f"run ledger: {e['scope']} run {e['run_id']}: {rel} was deleted")
            elif _sha(p) != h:
                findings.append(f"run ledger: {e['scope']} run {e['run_id']}: {rel} changed after it was recorded")
    listed = {(e["scope"], e["run_id"]) for e in entries}
    for scope, d in dirs.items():
        for rid in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if (scope, rid) not in listed:
                info.append(f"run ledger: {scope} run {rid} not recorded yet (added on the next scheduled run)")
    info.append(f"run ledger: {len(entries)} runs, head {entries[-1]['hash'][:16]}")
    return findings, info


# ------------------------------------------------------------------ checkpoint outside the state folder
def default_checkpoint_dir():
    """OF2_CHECKPOINT_DIR if set; for a non-default state (OF2_STATE_DIR, used by tests) a folder inside it;
    otherwise %LOCALAPPDATA%\\OnlyFunds\\shadow_checkpoint, outside the project and outside git. It holds only
    hashes, sizes and counts, never market data or credentials."""
    if os.environ.get("OF2_CHECKPOINT_DIR"):
        return os.environ["OF2_CHECKPOINT_DIR"]
    if os.environ.get("OF2_STATE_DIR"):
        return os.path.join(os.environ["OF2_STATE_DIR"], "_checkpoint")
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return os.path.join(base, "OnlyFunds", "shadow_checkpoint")


def _read_cp(checkpoint_dir):
    p = os.path.join(checkpoint_dir, CHECKPOINT)
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _prefix_sha(path, n):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read(n)).hexdigest()


def checkpoint_problems(state_root, checkpoint_dir, logs=True):
    """Differences between the state and the last checkpoint (empty list = consistent or no checkpoint yet)."""
    try:
        cp = _read_cp(checkpoint_dir).get("states", {}).get(os.path.abspath(state_root), {}).get("latest")
    except Exception as exc:
        return [f"checkpoint unreadable ({type(exc).__name__})"]
    if not cp:
        return []
    shadow = os.path.join(state_root, "shadow")
    out = []
    entries = _ledger_entries(shadow)
    n, head = cp["ledger_entries"], cp["ledger_head"]
    if n:
        if entries is None:
            out.append(f"run ledger deleted: the checkpoint of {cp['utc']} recorded {n} entries")
        elif len(entries) < n:
            out.append(f"run ledger truncated: {len(entries)} entries, the checkpoint of {cp['utc']} recorded {n}")
        elif entries[n - 1].get("hash") != head:
            out.append(f"run ledger rewritten: entry {n - 1} differs from the checkpoint of {cp['utc']}")
    if logs:
        for name, rec in cp.get("logs", {}).items():
            p = os.path.join(shadow, name)
            if not os.path.exists(p):
                out.append(f"{name} deleted (the checkpoint of {cp['utc']} recorded {rec['bytes']} bytes)")
            elif os.path.getsize(p) < rec["bytes"]:
                out.append(f"{name} truncated (shorter than at the checkpoint of {cp['utc']})")
            elif _prefix_sha(p, rec["bytes"]) != rec["sha256"]:
                out.append(f"{name} edited: its first {rec['bytes']} bytes differ from the checkpoint of {cp['utc']}")
    return out


def update_checkpoint(state_root, checkpoint_dir, now=None):
    """Record the current ledger head and log sizes. Refuses (returns the problems) while the state is
    inconsistent with the previous checkpoint, so an alteration is never absorbed into a new checkpoint."""
    probs = checkpoint_problems(state_root, checkpoint_dir)
    if probs:
        return probs
    shadow = os.path.join(state_root, "shadow")
    entries = _ledger_entries(shadow) or []
    logs = {}
    for name in APPEND_ONLY_LOGS:
        p = os.path.join(shadow, name)
        if os.path.exists(p):
            logs[name] = {"bytes": os.path.getsize(p), "sha256": _sha(p)}
    rec = {"utc": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"), "ledger_entries": len(entries),
           "ledger_head": entries[-1]["hash"] if entries else None, "logs": logs}
    data = _read_cp(checkpoint_dir)
    st = data.setdefault("states", {}).setdefault(os.path.abspath(state_root), {})
    st["latest"] = rec
    st["history"] = (st.get("history") or [])[-999:] + [{k: rec[k] for k in ("utc", "ledger_entries", "ledger_head")}]
    os.makedirs(checkpoint_dir, exist_ok=True)
    tmp = os.path.join(checkpoint_dir, CHECKPOINT + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, sort_keys=True)
    os.replace(tmp, os.path.join(checkpoint_dir, CHECKPOINT))
    return []


# ------------------------------------------------------------------ data quality
def snapshot_quality(csv_path):
    """Problems in one daily snapshot CSV (empty list = clean)."""
    import pandas as pd
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        return [f"unreadable ({type(exc).__name__})"]
    need = ["timestamp", "Open", "High", "Low", "Close", "Volume"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        return [f"missing columns {miss}"]
    if df.empty:
        return ["no rows"]
    out = []
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if ts.isna().any():
        out.append(f"{int(ts.isna().sum())} unparseable timestamps")
    px = df[["Open", "High", "Low", "Close"]].apply(pd.to_numeric, errors="coerce")
    vol = pd.to_numeric(df["Volume"], errors="coerce")
    if px.isna().any().any() or vol.isna().any():
        out.append("missing (NaN) prices or volume")
    if (px <= 0).any().any() or (vol < 0).any():
        out.append("non-positive prices or negative volume")
    tol = 1e-9 * px["High"].abs()
    bad = (px["High"] + tol < px[["Open", "Close"]].max(axis=1)) | (px["Low"] - tol > px[["Open", "Close"]].min(axis=1)) \
        | (px["Low"] > px["High"])
    if bad.any():
        out.append(f"{int(bad.sum())} inconsistent OHLC rows (first {df['timestamp'][bad].iloc[0]})")
    t = ts.dropna()
    if t.duplicated().any():
        out.append(f"{int(t.duplicated().sum())} duplicate days")
    if not t.is_monotonic_increasing:
        out.append("days out of order")
    gaps = t.sort_values().diff().dt.days.fillna(1)
    if (gaps > 1).any():
        out.append(f"{int((gaps - 1).clip(lower=0).sum())} missing days inside the series")
    if ((t - t.dt.normalize()) != pd.Timedelta(0)).any():
        out.append("timestamps not at 00:00 UTC (not daily candles)")
    return out


def _fresh(st):
    return isinstance(st, dict) and not st.get("stale") and not st.get("note")


# ------------------------------------------------------------------ main check
def check(state_root=None, now=None, pending_entry=None, checkpoint_dir=None):
    """`pending_entry`: the wrapper's journal entry for the run in progress (not yet written to the journal).
    `checkpoint_dir`: where the out-of-state checkpoint lives (None: checkpoint not checked)."""
    import forward_experimental as FE
    import forward_shadow as FS
    state_root = state_root or os.path.dirname(FS.STATE)
    now = now or datetime.now(timezone.utc)
    shadow, exp = os.path.join(state_root, "shadow"), os.path.join(state_root, "shadow_experimental")
    rep = {"ok": True, "checked_utc": now.isoformat(timespec="seconds"), "findings": [], "issues": [], "runs": 0,
           "snapshots_verified": 0, "journal": {}, "experimental_runs": 0}

    def bad(msg, key, level=CRITICAL):
        rep["ok"] = False
        rep["findings"].append(msg)
        rep["issues"].append({"key": key, "level": level, "message": msg})

    def info(msg):
        rep.setdefault("info", []).append(msg)

    fz, prob = FS.load_frozen()
    for p in prob:
        bad(f"frozen T50/RS3: {p}", f"freeze:frozen:{p}")
    rep["frozen_fingerprints"] = fz.get("fingerprints")
    if os.path.exists(FE.EXP_FROZEN):
        efz, eprob = FE.load_frozen()
        for p in eprob:
            bad(f"experimental X1/X2: {p}", f"freeze:experimental:{p}")
        rep["experimental_fingerprints"] = efz.get("fingerprints")

    # 2) frozen runs
    runs_dir = os.path.join(shadow, "runs")
    frozen_runs = sorted(os.listdir(runs_dir)) if os.path.isdir(runs_dir) else []
    records = []
    for rid in frozen_runs:
        p = os.path.join(runs_dir, rid, "run.json")
        if not os.path.exists(p):
            bad(f"run {rid}: no run.json (crashed or killed before it finished)", f"run:{rid}:no_record")
            continue
        rep["runs"] += 1
        try:
            with open(p, encoding="utf-8") as f:
                rec = json.load(f)
        except Exception as exc:
            bad(f"run {rid}: run.json unreadable ({type(exc).__name__})", f"run:{rid}:unreadable")
            continue
        records.append((rid, rec))
        for sym, st in (rec.get("data_status") or {}).items():
            if not isinstance(st, dict) or "sha256" not in st:
                continue
            csv = os.path.join(runs_dir, rid, "data", f"{sym}_1d.csv")
            if not os.path.exists(csv):
                bad(f"run {rid}: snapshot {sym} missing", f"run:{rid}:snapshot_missing:{sym}")
                continue
            if _sha(csv) != st["sha256"]:
                bad(f"run {rid}: snapshot {sym} does not match its recorded hash", f"run:{rid}:snapshot_hash:{sym}")
            else:
                rep["snapshots_verified"] += 1
        superseded = False
        if rec.get("frozen_fingerprints") and rep.get("frozen_fingerprints") and \
                rec["frozen_fingerprints"] != rep["frozen_fingerprints"]:
            frozen_at = fz.get("frozen_at_utc", "")[:19].replace("-", "").replace(":", "")   # 20261101T122154
            if rid[:15] < frozen_at and not str(rec.get("status", "")).startswith("ok"):
                superseded = True
                info(f"run {rid}: made before the current freeze ({fz['frozen_at_utc']}) with the superseded freeze; "
                     "evaluated nothing; kept as history")
            else:
                bad(f"run {rid}: recorded fingerprints differ from the freeze file", f"run:{rid}:fingerprints")
        status = str(rec.get("status", ""))
        if not superseded and not status.startswith(NORMAL_STATUS):
            bad(f"run {rid}: status {status[:90]!r}", f"run:{rid}:status")
        if rec.get("anomalies"):
            bad(f"run {rid}: anomalies {rec['anomalies']}", f"run:{rid}:anomalies")

    # 5) data quality of the latest snapshot, and coins that dropped out since the previous run
    with_data = [(rid, rec) for rid, rec in records if rec.get("data_status")]
    if with_data:
        rid, rec = with_data[-1]
        rep["latest_snapshot"] = rid
        for sym, st in sorted(rec["data_status"].items()):
            if isinstance(st, dict) and "sha256" in st:
                for prob_ in snapshot_quality(os.path.join(runs_dir, rid, "data", f"{sym}_1d.csv")):
                    kind = re.sub(r"\d+", "#", prob_.split(" (first")[0])
                    bad(f"run {rid}: snapshot {sym}: {prob_}", f"data_quality:{sym}:{kind}")
        if len(with_data) >= 2:
            prev_id, prev_rec = with_data[-2]
            for sym, st in sorted(rec["data_status"].items()):
                if _fresh(prev_rec["data_status"].get(sym)) and not _fresh(st):
                    why = st if not isinstance(st, dict) else ("stale" if st.get("stale") else st.get("note"))
                    bad(f"run {rid}: {sym} was fresh in run {prev_id} and is now {why!s:.80} "
                        "(check for a data outage or a delisting)", f"dropout:{sym}:{rid}", WARNING)
        unavailable = sorted(s for s, st in rec["data_status"].items() if not _fresh(st))
        if unavailable:
            info(f"coins not fresh in the latest snapshot (known delistings are expected): {', '.join(unavailable)}")

    # 3) journal
    jp = os.path.join(shadow, "task_journal.jsonl")
    j = {"entries": 0, "ok": 0, "refused_overlap": 0, "crashed": 0, "other_exit": 0, "last": None, "warnings": [],
         "environment_changes": []}
    entries = []
    if os.path.exists(jp):
        with open(jp, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    entries.append(json.loads(line))
                except ValueError:
                    bad(f"journal: line {n} is not valid JSON", f"journal:invalid:{n}")
    if pending_entry is not None:
        entries.append(pending_entry)
    last_env = None
    for i, e in enumerate(entries):
        j["entries"] += 1
        code, when = e.get("exit_code"), e.get("started_utc")
        if code == 0:
            j["ok"] += 1
        elif code == 9:
            j["refused_overlap"] += 1
        elif code == 10 or "CRASHED" in str(e.get("status")):
            j["crashed"] += 1
            bad(f"journal: crash at {when}: {e.get('status')}", f"journal:crash:{when}")
        else:
            j["other_exit"] += 1
            bad(f"journal: exit {code} at {when}: {e.get('status')}", f"journal:exit:{when}")
        ex = e.get("experimental") or {}
        if ex.get("status") == "CRASHED":
            bad(f"journal: experimental step crashed at {when}", f"journal:experimental:{when}")
        elif ex.get("ran") and ex.get("exit_code") not in (0, None):
            bad(f"journal: experimental step exit {ex.get('exit_code')} at {when}: {ex.get('status')}",
                f"journal:experimental:{when}")
        if e.get("env"):
            if last_env is not None and e["env"] != last_env:
                change = {k: [last_env.get(k), e["env"].get(k)] for k in sorted(set(last_env) | set(e["env"]))
                          if last_env.get(k) != e["env"].get(k)}
                j["environment_changes"].append({"at": when, "changes": change})
                if i == len(entries) - 1:
                    bad(f"journal: Python environment changed in the latest run {change}; results may differ "
                        "(DATA_REVISION would show it)", f"env:{when}", WARNING)
            last_env = e["env"]
        if code != 9:
            j["last"] = {k: e.get(k) for k in ("started_utc", "exit_code", "status", "evaluated_through")}
            j["warnings"] = e.get("data_warnings") or []
    rep["journal"] = j

    # 4) missed runs and evaluation lag
    done = sorted({d for d in (_day(e.get("started_utc")) for e in entries if e.get("exit_code") == 0) if d})
    if done:
        last_due = (now - SCHEDULE_UTC - GRACE_AFTER_SCHEDULE).date()
        day = done[0]
        while day <= last_due:
            if day not in done:
                bad(f"missed runs: no successful tracker run on UTC day {day}", f"missed:{day}")
            day += timedelta(days=1)
        rep["last_successful_run_day"] = str(done[-1])
        for e in reversed(entries):
            if e.get("exit_code") == 0 and e.get("evaluated_through") and _day(e.get("started_utc")):
                lag = (_day(e["started_utc"]) - datetime.fromisoformat(e["evaluated_through"]).date()).days
                rep["evaluation_lag_days"] = lag
                if lag > 1:
                    bad(f"evaluation lags: run on {_day(e['started_utc'])} evaluated only through "
                        f"{e['evaluated_through']} ({lag} days)", "evaluation_lag", WARNING)
                break

    # 6) experimental runs
    er = os.path.join(exp, "runs")
    for rid in sorted(os.listdir(er)) if os.path.isdir(er) else []:
        rep["experimental_runs"] += 1
        p = os.path.join(er, rid, "run.json")
        if not os.path.exists(p):
            bad(f"experimental run {rid}: no run.json", f"exp:{rid}:no_record")
            continue
        with open(p, encoding="utf-8") as f:
            erec = json.load(f)
        ref = erec.get("frozen_run")
        if ref and ref not in frozen_runs:
            bad(f"experimental run {rid} references a frozen run that does not exist ({ref})", f"exp:{rid}:reference")
        if erec.get("status") not in (None, "ok"):
            bad(f"experimental run {rid}: status {str(erec.get('status'))[:90]!r}", f"exp:{rid}:status")
        if erec.get("anomalies"):
            bad(f"experimental run {rid}: anomalies {erec['anomalies']}", f"exp:{rid}:anomalies")
    if os.path.abspath(exp).startswith(os.path.abspath(shadow) + os.sep):
        bad("experimental state lives inside the frozen state directory", "exp:location")

    # 7) ledger and checkpoint
    lf, li = verify_ledger(state_root)
    for m in lf:
        bad(m, f"ledger:{m}")
    for m in li:
        info(m)
    if checkpoint_dir is not None:
        rep["checkpoint_dir"] = checkpoint_dir
        for m in checkpoint_problems(state_root, checkpoint_dir):
            bad(f"checkpoint: {m}", f"checkpoint:{m.split(':')[0]}")
        if not os.path.exists(os.path.join(checkpoint_dir, CHECKPOINT)):
            info("no checkpoint outside the state folder yet (the scheduled wrapper writes it after each run)")

    # 8) no order capability
    for rel in TRACKING_FILES:
        src = open(os.path.join(ROOT, rel), encoding="utf-8").read()
        for pat in ORDER_PATTERNS:
            if re.search(pat, src):
                bad(f"{rel}: order/credential pattern {pat!r} found", f"order:{rel}:{pat}")
    rep["critical"] = sum(i["level"] == CRITICAL for i in rep["issues"])
    rep["warning_count"] = sum(i["level"] == WARNING for i in rep["issues"])
    return rep


def main():
    rep = check(checkpoint_dir=default_checkpoint_dir())
    print(json.dumps(rep, indent=1, default=str))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
