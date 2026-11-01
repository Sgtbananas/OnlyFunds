"""Read-only health check of forward shadow tracking (frozen T50/RS3 and experimental X1/X2). Research only.

Changes nothing: it only reads state/ and the freeze files, and prints (or returns) a list of findings.

Checks
1. Freeze files: the frozen T50/RS3 implementation and spec fingerprints, and the experimental X1/X2 ones, match.
2. Every stored frozen run: run.json is readable, and every snapshot CSV still matches its recorded SHA-256
   (historical snapshots intact).
3. Run journal: every scheduled or manual invocation has an exit code; lists crashes, refusals (overlap lock) and
   data warnings. A crash is reported, not hidden.
4. Experimental runs reference an existing frozen run, and never live inside the frozen state directory.
5. No exchange-order capability: no order, signing or credential code in the tracking files.

    python research/shadow_health.py
"""
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

TRACKING_FILES = ["research/forward_shadow.py", "research/forward_experimental.py", "research/shadow_task.py",
                  "research/portfolio.py", "research/portfolio_variants.py", "research/round1.py"]
ORDER_PATTERNS = [r"place_order", r"create_order", r"cancel_order", r"/api/v\d/order", r"hmac\.new", r"X-MBX-APIKEY",
                  r"CoinExClient", r"requests\.post", r"API_SECRET\s*="]


def check(state_root=None):
    import forward_experimental as FE
    import forward_shadow as FS
    state_root = state_root or os.path.dirname(FS.STATE)
    shadow, exp = os.path.join(state_root, "shadow"), os.path.join(state_root, "shadow_experimental")
    rep = {"ok": True, "findings": [], "runs": 0, "snapshots_verified": 0, "journal": {}, "experimental_runs": 0}

    def bad(msg):
        rep["ok"] = False
        rep["findings"].append(msg)

    fz, prob = FS.load_frozen()
    for p in prob:
        bad(f"frozen T50/RS3: {p}")
    rep["frozen_fingerprints"] = fz.get("fingerprints")
    if os.path.exists(FE.EXP_FROZEN):
        efz, eprob = FE.load_frozen()
        for p in eprob:
            bad(f"experimental X1/X2: {p}")
        rep["experimental_fingerprints"] = efz.get("fingerprints")

    runs_dir = os.path.join(shadow, "runs")
    frozen_runs = sorted(os.listdir(runs_dir)) if os.path.isdir(runs_dir) else []
    for rid in frozen_runs:
        p = os.path.join(runs_dir, rid, "run.json")
        if not os.path.exists(p):
            bad(f"run {rid}: no run.json")
            continue
        rep["runs"] += 1
        with open(p, encoding="utf-8") as f:
            rec = json.load(f)
        for sym, st in (rec.get("data_status") or {}).items():
            if not isinstance(st, dict) or "sha256" not in st:
                continue
            csv = os.path.join(runs_dir, rid, "data", f"{sym}_1d.csv")
            if not os.path.exists(csv):
                bad(f"run {rid}: snapshot {sym} missing")
                continue
            with open(csv, "rb") as f:
                if hashlib.sha256(f.read()).hexdigest() != st["sha256"]:
                    bad(f"run {rid}: snapshot {sym} does not match its recorded hash")
                else:
                    rep["snapshots_verified"] += 1
        if rec.get("frozen_fingerprints") and rep.get("frozen_fingerprints") and \
                rec["frozen_fingerprints"] != rep["frozen_fingerprints"]:
            frozen_at = fz.get("frozen_at_utc", "")[:19].replace("-", "").replace(":", "")   # 20261101T122154
            if rid[:15] < frozen_at and not str(rec.get("status", "")).startswith("ok"):
                rep.setdefault("info", []).append(f"run {rid}: made before the current freeze ({fz['frozen_at_utc']}) "
                                                  "with the superseded freeze; evaluated nothing; kept as history")
            else:
                bad(f"run {rid}: recorded fingerprints differ from the freeze file")

    jp = os.path.join(shadow, "task_journal.jsonl")
    j = {"entries": 0, "ok": 0, "refused_overlap": 0, "crashed": 0, "other_exit": 0, "last": None, "warnings": []}
    if os.path.exists(jp):
        with open(jp, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                e = json.loads(line)
                j["entries"] += 1
                code = e.get("exit_code")
                if code == 0:
                    j["ok"] += 1
                elif code == 9:
                    j["refused_overlap"] += 1
                elif code == 10 or "CRASHED" in str(e.get("status")):
                    j["crashed"] += 1
                    bad(f"journal: crash at {e.get('started_utc')}: {e.get('status')}")
                else:
                    j["other_exit"] += 1
                    bad(f"journal: exit {code} at {e.get('started_utc')}: {e.get('status')}")
                ex = e.get("experimental") or {}
                if ex.get("status") == "CRASHED":
                    bad(f"journal: experimental step crashed at {e.get('started_utc')}")
                j["last"] = {k: e.get(k) for k in ("started_utc", "exit_code", "status", "evaluated_through")}
                j["warnings"] = e.get("data_warnings") or []
    rep["journal"] = j

    er = os.path.join(exp, "runs")
    for rid in sorted(os.listdir(er)) if os.path.isdir(er) else []:
        rep["experimental_runs"] += 1
        p = os.path.join(er, rid, "run.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                ref = json.load(f).get("frozen_run")
            if ref and ref not in frozen_runs:
                bad(f"experimental run {rid} references a frozen run that does not exist ({ref})")
    if os.path.abspath(exp).startswith(os.path.abspath(shadow) + os.sep):
        bad("experimental state lives inside the frozen state directory")

    for rel in TRACKING_FILES:
        src = open(os.path.join(ROOT, rel), encoding="utf-8").read()
        for pat in ORDER_PATTERNS:
            if re.search(pat, src):
                bad(f"{rel}: order/credential pattern {pat!r} found")
    return rep


def main():
    rep = check()
    print(json.dumps(rep, indent=1, default=str))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
