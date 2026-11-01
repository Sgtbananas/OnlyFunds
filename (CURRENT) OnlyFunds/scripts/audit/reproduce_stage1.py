"""Audit: rerun validation stage 1 into a TEMPORARY folder and compare with the committed results.

Never touches docs/validation/ (committed results) and never runs the final TEST stage.

    python scripts/audit/reproduce_stage1.py --out docs/audit/stage1_reproduction.json
"""
import argparse
import json
import os
import re
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from scripts import real_market_validation as R  # noqa: E402


def table_rows(md_text):
    """Numeric cells of the stage-1 summary table, keyed by (candidate, period)."""
    sect = md_text.split("### Summary")[1].split("### Walk-forward")[0]
    rows = {}
    for line in sect.splitlines():
        if line.startswith("|") and not line.startswith("|---") and "candidate" not in line:
            cells = [c.strip() for c in line.strip("|").split("|")]
            rows[(cells[0], cells[1])] = cells[2:15]
    return rows


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    committed_dir = os.path.join(ROOT, "docs", "validation")
    tmp = tempfile.mkdtemp(prefix="of2_stage1_repro_")
    R.OUT_DIR, R.FROZEN = tmp, os.path.join(tmp, "frozen_decisions.json")
    args = type("A", (), {"symbols": json.load(open(os.path.join(committed_dir, "frozen_decisions.json")))["symbols"]})()
    R.stage_devval(args)
    old_f = json.load(open(os.path.join(committed_dir, "frozen_decisions.json"), encoding="utf-8"))
    new_f = json.load(open(R.FROZEN, encoding="utf-8"))
    old_t = table_rows(open(os.path.join(committed_dir, "stage1_dev_val.md"), encoding="utf-8").read())
    new_t = table_rows(open(os.path.join(tmp, "stage1_dev_val.md"), encoding="utf-8").read())
    old_wf = re.findall(r"\*\*(\w+)\*\*: (\d/4)", open(os.path.join(committed_dir, "stage1_dev_val.md"), encoding="utf-8").read())
    new_wf = re.findall(r"\*\*(\w+)\*\*: (\d/4)", open(os.path.join(tmp, "stage1_dev_val.md"), encoding="utf-8").read())
    result = {
        "temp_dir": tmp,
        "frozen_candidates_identical": old_f["candidates"] == new_f["candidates"],
        "protocol_fingerprint_identical": old_f["protocol_sha256"] == new_f["protocol_sha256"],
        "summary_rows_compared": len(old_t),
        "summary_rows_identical": sum(1 for k in old_t if new_t.get(k) == old_t[k]),
        "summary_mismatches": [str(k) for k in old_t if new_t.get(k) != old_t[k]],
        "walk_forward_identical": old_wf == new_wf,
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
