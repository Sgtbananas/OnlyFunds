"""Run the pre-registered real-market validation (docs/VALIDATION_PROTOCOL.md).

    python scripts/fetch_history.py                         # once: download data (checksummed)
    python scripts/real_market_validation.py --stage devval # DEV tuning, VAL checks, walk-forward, ML; freezes decisions
    python scripts/real_market_validation.py --stage test   # ONE evaluation of the frozen candidates on TEST

The TEST stage refuses to run without the frozen file, or if the protocol changed since freezing,
and refuses to overwrite an existing TEST result (use --rerun-test only to reproduce, which is
recorded in the output).
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd  # noqa: E402

from core import validation as V  # noqa: E402
from core.history import DEFAULT_UNIVERSE, load_dataset  # noqa: E402
from utils.config import APP_ROOT, build_settings, load_config  # noqa: E402

OUT_DIR = os.path.join(APP_ROOT, "docs", "validation")
FROZEN = os.path.join(OUT_DIR, "frozen_decisions.json")
PROTOCOL = os.path.join(APP_ROOT, "docs", "VALIDATION_PROTOCOL.md")


def fmt(df: pd.DataFrame) -> str:
    if df is None or len(df) == 0:
        return "_none_"
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "nan" if pd.isna(v) else f"{v:.3f}")
    return ("| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns) + "\n" +
            "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in d.itertuples(index=False)))


def summary_row(name, period, p, passed=None, reasons=None):
    return {"candidate": name, "period": period, "trades": p["trades"], "expectancy_r": p["expectancy_r"],
            "ci95_low_r": p["ci_low"], "profit_factor": p["profit_factor"], "win_rate_pct": p["win_rate_pct"],
            "avg_win_r": p["avg_win_r"], "avg_loss_r": p["avg_loss_r"], "breadth": p["breadth"],
            "exp_r_wo_best": p["exp_r_without_best_coin"], "mean_net_ret_pct": p["mean_net_return_pct"],
            "mean_buy_hold_pct": p["mean_buy_hold_pct"], "worst_dd_pct": p["worst_max_dd_pct"],
            **({"criteria": "PASS" if passed else "; ".join(reasons)} if passed is not None else {})}


def detail(title, p):
    out = f"\n#### {title}\n\nPer coin:\n\n{fmt(p['per_coin'])}\n"
    if len(p.get("by_regime", [])):
        out += f"\nBy regime at entry:\n\n{fmt(p['by_regime'].reset_index())}\n"
        out += f"\nBy year of entry:\n\n{fmt(p['by_year'].reset_index())}\n"
    return out


def load_data(symbols):
    return {s: load_dataset(s, "1h") for s in symbols}


def base_config(fee=None):
    bt, _ = build_settings(load_config(), "Conservative", {"interval": "1h"})
    bt.initial_capital = 1000.0
    if fee is not None:
        bt.fee_rate = fee
    return bt


def stage_devval(a):
    # Audit fix P2: never overwrite frozen v1 results. New validations use scripts/validation_run.py,
    # which writes each run to its own immutable directory.
    if os.path.exists(FROZEN):
        print(f"REFUSED: {FROZEN} already exists (frozen v1 results are immutable). "
              "Use scripts/validation_run.py for new validation runs.")
        return 2
    t0 = time.time()
    data = load_data(a.symbols)
    base = base_config()
    rows, details, frozen = [], "", {"protocol_sha256": V.protocol_fingerprint(PROTOCOL),
                                     "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                     "symbols": a.symbols, "candidates": {}}
    wf_tables = ""
    for strat in V.STRATEGIES:
        print(f"[{strat}] fixed DEV/VAL ...", flush=True)
        for per in ("DEV", "VAL"):
            p = V.pooled(V.run_period(data, base, strat, None, *V.PERIODS[per]))
            ok, why = V.criteria(p)
            rows.append(summary_row(f"{strat}/fixed", per, p, ok if per == "VAL" else None, why))
            if per == "VAL":
                details += detail(f"{strat}/fixed, VAL", p)
        print(f"[{strat}] tuning on DEV ...", flush=True)
        dev_params, dev_grid = V.tune(data, base, strat, *V.PERIODS["DEV"])
        if dev_params:
            p = V.pooled(V.run_period(data, base, strat, dev_params, *V.PERIODS["VAL"]))
            ok, why = V.criteria(p)
            rows.append(summary_row(f"{strat}/tuned{dev_params}", "VAL", p, ok, why))
        print(f"[{strat}] walk-forward ...", flush=True)
        wf = V.walk_forward(data, base, strat)
        wf_ok, wf_msg = V.walk_forward_pass(wf)
        wf_tables += f"\n**{strat}**: {wf_msg}\n\n{fmt(wf.assign(params=wf['params'].astype(str)))}\n"
        print(f"[{strat}] re-tuning on DEV+VAL for TEST ...", flush=True)
        final_params, _ = V.tune(data, base, strat, V.PERIODS["DEV"][0], V.PERIODS["VAL"][1])
        print(f"[{strat}] ML (train DEV, judge VAL) ...", flush=True)
        ml_note, ml_carry = "", False
        try:
            bundle = V.train_ml(data, base, strat)
            rep = bundle["report"]
            p_ml = V.pooled(V.run_period(data, base, strat, None, *V.PERIODS["VAL"], ml_bundle=bundle))
            p_no = next(r for r in rows if r["candidate"] == f"{strat}/fixed" and r["period"] == "VAL")
            ok, why = V.criteria(p_ml)
            rows.append(summary_row(f"{strat}/fixed+ML", "VAL", p_ml, ok, why))
            ml_carry = bool(p_ml["trades"] > 0 and p_ml["expectancy_r"] > p_no["expectancy_r"])
            ml_note = (f"internal DEV validation: AUC {rep['validation_auc']:.3f}, useful={rep['useful']}; "
                       f"VAL expectancy with ML {p_ml['expectancy_r']:+.3f}R ({p_ml['trades']} trades) vs without "
                       f"{p_no['expectancy_r']:+.3f}R ({p_no['trades']} trades) -> carried to TEST: {ml_carry}")
            if ml_carry:
                import joblib
                os.makedirs(OUT_DIR, exist_ok=True)
                joblib.dump(bundle, os.path.join(OUT_DIR, f"ml_{strat}.pkl"))
        except ValueError as exc:
            ml_note = f"ML not trainable: {exc}"
        frozen["candidates"][strat] = {"fixed": None, "tuned_dev_val": final_params, "walk_forward_pass": wf_ok,
                                       "walk_forward": wf_msg, "ml_carry": ml_carry, "ml_note": ml_note}
    summary = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(FROZEN, "w", encoding="utf-8") as f:
        json.dump(frozen, f, indent=2, default=str)
    text = (f"## Stage 1: development and validation (run {frozen['frozen_at_utc']}, {time.time() - t0:.0f}s)\n\n"
            f"Coins: {', '.join(a.symbols)}. Costs: fee {base.fee_rate:.2%}/side, slippage {base.slippage_pct:.2%}, "
            f"stop slippage {base.stop_slippage_pct:.2%}; Conservative risk; 1h; 1,000 USDT per coin.\n\n"
            f"### Summary (criteria judged on VAL; DEV is in-sample)\n\n{fmt(summary)}\n\n"
            f"### Walk-forward (tuned on the 2 prior years, scored on the year)\n{wf_tables}\n"
            f"### ML\n\n" + "\n".join(f"- **{s}**: {c['ml_note']}" for s, c in frozen["candidates"].items()) +
            f"\n\n### Frozen for TEST\n\n```json\n{json.dumps(frozen['candidates'], indent=1, default=str)}\n```\n"
            f"\n### VAL detail\n{details}\n")
    with open(os.path.join(OUT_DIR, "stage1_dev_val.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text[:3000])
    return 0


def stage_test(a):
    if not os.path.exists(FROZEN):
        print("Run --stage devval first (frozen decisions missing).")
        return 2
    with open(FROZEN, encoding="utf-8") as f:
        frozen = json.load(f)
    if frozen["protocol_sha256"] != V.protocol_fingerprint(PROTOCOL):
        print("Protocol changed after freezing; refusing to run the final test.")
        return 2
    out_md = os.path.join(OUT_DIR, "stage2_test.md")
    if os.path.exists(out_md) and not a.rerun_test:
        print(f"{out_md} exists: the final test has already been run once. Use --rerun-test to reproduce.")
        return 2
    data = load_data(frozen["symbols"])
    rows, details = [], ""
    for fee_label, fee in (("CoinEx-level costs", None), ("info only: 0.1% fee", 0.001)):
        base = base_config(fee)
        for strat, c in frozen["candidates"].items():
            variants = [("fixed", None, None), ("tuned", c["tuned_dev_val"], None)]
            if c["ml_carry"]:
                import joblib
                variants.append(("fixed+ML", None, joblib.load(os.path.join(OUT_DIR, f"ml_{strat}.pkl"))))
            for vname, params, bundle in variants:
                if vname == "tuned" and not params:
                    continue
                p = V.pooled(V.run_period(data, base, strat, params, *V.PERIODS["TEST"], ml_bundle=bundle))
                ok, why = V.criteria(p)
                wf_ok = c["walk_forward_pass"] or vname == "fixed+ML"
                if not wf_ok:
                    why = why + [c["walk_forward"]]
                rows.append({"costs": fee_label, **summary_row(f"{strat}/{vname}", "TEST", p, ok and wf_ok, why)})
                if fee is None:
                    details += detail(f"{strat}/{vname}, TEST", p)
    summary = pd.DataFrame(rows)
    text = (f"## Stage 2: final test, run once ({datetime.now(timezone.utc).isoformat(timespec='seconds')}"
            f"{', REPRODUCTION RUN' if a.rerun_test else ''})\n\n"
            f"Frozen decisions from {frozen['frozen_at_utc']}; protocol fingerprint verified.\n\n"
            f"{fmt(summary)}\n\n### TEST detail (CoinEx-level costs)\n{details}\n")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(text)
    print(text[:4000])
    return 0


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles can't print the report's symbols otherwise
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["devval", "test"], required=True)
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_UNIVERSE)
    ap.add_argument("--rerun-test", action="store_true")
    a = ap.parse_args(argv)
    return stage_devval(a) if a.stage == "devval" else stage_test(a)


if __name__ == "__main__":
    sys.exit(main())
