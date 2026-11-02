"""Validation framework v2 (fixes audit issues P1-P4). See docs/VALIDATION_PROTOCOL_V2.md.

    python scripts/validation_run.py stage1 --plan docs/validation_plans/<plan>.json
    python scripts/validation_run.py stage2 --plan docs/validation_plans/<plan>.json --run-id <id>
    python scripts/validation_run.py list   --plan docs/validation_plans/<plan>.json

- Stage 1 always creates a new, immutable run directory (P2) with a provenance manifest (P1),
  per-candidate eligibility decisions (P3) and ML promotion decisions (P4).
- Stage 2 evaluates ONLY eligible, frozen candidates, refuses if the protocol, plan, implementation,
  configuration or data changed since Stage 1, refuses a TEST window that was already consumed,
  and never overwrites anything. If nothing is eligible, TEST is not evaluated and not consumed.
- The v1 evaluation (scripts/real_market_validation.py, docs/validation/*.md) is left untouched.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core import validation as V  # noqa: E402
from core.core_signals import STRATEGIES  # noqa: E402
from core.eligibility import DEFAULT_RULES, candidate_eligibility, ml_promotion  # noqa: E402
from core.history import load_dataset  # noqa: E402
from core.provenance import (build_manifest, dataset_provenance, diff_components, fingerprint_components,  # noqa: E402
                             git_state, sha256_file, validation_fingerprint)
from core.run_store import ImmutableError, RunStore, write_once  # noqa: E402
from utils.config import build_settings, load_config  # noqa: E402


class RefusedError(RuntimeError):
    pass


# ------------------------------------------------------------------ plan & setup
def load_plan(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        plan = json.load(f)
    for k in ("name", "protocol", "exchange", "interval", "symbols", "periods", "walk_forward", "strategies"):
        if k not in plan:
            raise RefusedError(f"plan is missing '{k}'")
    for p in ("DEV", "VAL", "TEST"):
        if p not in plan["periods"]:
            raise RefusedError(f"plan.periods is missing {p}")
    s = {p: [pd.Timestamp(x, tz="UTC") for x in plan["periods"][p]] for p in ("DEV", "VAL", "TEST")}
    if not (s["DEV"][0] < s["DEV"][1] <= s["VAL"][0] < s["VAL"][1] <= s["TEST"][0] < s["TEST"][1]):
        raise RefusedError("periods must be chronological and non-overlapping: DEV < VAL < TEST")
    for fold in plan["walk_forward"]:
        tr0, tr1, te0, te1 = (pd.Timestamp(x, tz="UTC") for x in fold)
        if not (tr0 < tr1 <= te0 < te1 <= s["TEST"][0]):
            raise RefusedError(f"walk-forward fold {fold} is not chronological or touches TEST")
    return plan


def base_config(plan: dict):
    bt, _ = build_settings(load_config(), plan.get("profile", "Conservative"), {"interval": plan["interval"]})
    bt.initial_capital = float(plan.get("initial_capital", 1000.0))
    for k, v in (plan.get("costs") or {}).items():
        setattr(bt, k, v)
    return bt


def load_data(plan: dict) -> dict:
    return {s: load_dataset(s, plan["interval"], plan["exchange"], plan.get("data_root")) for s in plan["symbols"]}


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def current_components(plan: dict, bt) -> tuple:
    data = dataset_provenance(plan["symbols"], plan["interval"], plan["exchange"], plan.get("data_root"))
    bad = [s for s, d in data.items() if not d["matches_manifest"]]
    if bad:
        raise RefusedError(f"datasets do not match their manifests: {bad}")
    return data, fingerprint_components(resolve(plan["protocol"]), plan, data, bt.as_dict())


def store_for(plan: dict) -> RunStore:
    return RunStore(resolve(plan["runs_root"])) if plan.get("runs_root") else RunStore()


def md_table(df: pd.DataFrame) -> str:
    if df is None or len(df) == 0:
        return "_none_"
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "nan" if pd.isna(v) else f"{v:.3f}")
    return ("| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns) + "\n" +
            "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in d.itertuples(index=False)))


def walk_forward(data, bt, strat, folds, grid):
    rows = []
    for tr0, tr1, te0, te1 in folds:
        params, _ = tune(data, bt, strat, tr0, tr1, grid)
        p = V.pooled(V.run_period(data, bt, strat, params, te0, te1)) if params else None
        rows.append({"test": f"{te0}..{te1}", "params": params, "trades": p["trades"] if p else 0,
                     "expectancy_r": p["expectancy_r"] if p else float("nan")})
    return pd.DataFrame(rows)


def tune(data, bt, strat, start, end, grid):
    old = V.GRID
    try:
        V.GRID = grid
        return V.tune(data, bt, strat, start, end)
    finally:
        V.GRID = old


def trade_r(results) -> np.ndarray:
    parts = [r.trades["r_multiple"].to_numpy(float) for r in results.values() if len(r.trades)]
    return np.concatenate(parts) if parts else np.array([])


# ------------------------------------------------------------------ stage 1
def stage1(plan_path: str, allow_dirty: bool = False, supersedes: str = None, reason: str = "",
           argv=None) -> str:
    plan = load_plan(plan_path)
    store = store_for(plan)
    store.check_test_window_unused(*plan["periods"]["TEST"], plan["symbols"], plan["interval"])
    git = git_state()
    if git["dirty"] and not allow_dirty:
        raise RefusedError("working tree has uncommitted changes; commit first (or --allow-dirty, which is recorded)")
    bt = base_config(plan)
    data_prov, comps = current_components(plan, bt)
    data = load_data(plan)
    rules = {**DEFAULT_RULES, **(plan.get("rules") or {})}
    grid = plan.get("grid", V.GRID)
    run_id = store.create_run(plan["name"], supersedes, reason)
    out = store.stage_dir(plan["name"], run_id, "stage1", create=True)
    dev, val = plan["periods"]["DEV"], plan["periods"]["VAL"]
    decisions, rows, models = {}, [], {}
    for strat in plan["strategies"]:
        wf = walk_forward(data, bt, strat, plan["walk_forward"], grid)
        fixed_res = V.run_period(data, bt, strat, None, *val)
        p_fixed = V.pooled(fixed_res)
        decisions[f"{strat}/fixed"] = {"strategy": strat, "variant": "fixed", "params": None,
                                       **candidate_eligibility(p_fixed, wf, rules)}
        dev_params, _ = tune(data, bt, strat, *dev, grid)
        if dev_params:
            p_tuned = V.pooled(V.run_period(data, bt, strat, dev_params, *val))
            final_params, _ = tune(data, bt, strat, dev[0], val[1], grid)
            decisions[f"{strat}/tuned"] = {"strategy": strat, "variant": "tuned", "params": final_params,
                                           "params_tuned_on_dev": dev_params,
                                           **candidate_eligibility(p_tuned, wf, rules)}
            rows.append(("tuned", strat, p_tuned))
        else:
            decisions[f"{strat}/tuned"] = {"strategy": strat, "variant": "tuned", "params": None, "eligible": False,
                                           "reasons": [f"no parameter set reached {rules['min_trades']} trades on DEV"],
                                           "evidence": {}}
        rows.append(("fixed", strat, p_fixed))
        if plan.get("ml", True):
            try:
                bundle = _train_ml_on(data, bt, strat, dev)  # trained on this plan's DEV period only
                ml_res = V.run_period(data, bt, strat, None, *val, ml_bundle=bundle)
                p_ml = V.pooled(ml_res)
                ml_elig = candidate_eligibility(p_ml, wf, rules)
                promo = ml_promotion(p_fixed, p_ml, trade_r(fixed_res), trade_r(ml_res), bundle["report"], ml_elig, rules)
                decisions[f"{strat}/fixed+ML"] = {"strategy": strat, "variant": "fixed+ML", "params": None,
                                                  "eligible": bool(promo["promoted"] and ml_elig["eligible"]),
                                                  "reasons": promo["reasons"], "evidence": promo["evidence"],
                                                  "p3": ml_elig}
                rows.append(("fixed+ML", strat, p_ml))
                if promo["promoted"]:
                    path = os.path.join(out, f"ml_{strat}.pkl")
                    import io
                    buf = io.BytesIO()
                    joblib.dump(bundle, buf)
                    write_once(path, buf.getvalue(), binary=True)
                    models[strat] = sha256_file(path)
            except ValueError as exc:
                decisions[f"{strat}/fixed+ML"] = {"strategy": strat, "variant": "fixed+ML", "eligible": False,
                                                  "reasons": [f"ML not trainable: {exc}"], "evidence": {}}
    manifest = build_manifest(run_id, "stage1", resolve(plan["protocol"]), plan, data_prov, bt.as_dict(),
                              {k: STRATEGIES[k] for k in STRATEGIES}, models,
                              extra={"allow_dirty": allow_dirty, "rules": rules}, argv=argv)
    if manifest["validation_fingerprint"] != validation_fingerprint(comps):
        raise RefusedError("inputs changed while Stage 1 was running; result discarded")
    write_once(os.path.join(out, "provenance.json"), manifest)
    write_once(os.path.join(out, "eligibility.json"), decisions)
    summary = pd.DataFrame([{"candidate": f"{s}/{v}", "trades": p["trades"], "expectancy_r": p["expectancy_r"],
                             "ci95_low_r": p["ci_low"], "profit_factor": p["profit_factor"],
                             "breadth": p["breadth"], "worst_dd_pct": p["worst_max_dd_pct"],
                             "eligible": decisions[f"{s}/{v}"]["eligible"],
                             "reasons": "; ".join(decisions[f"{s}/{v}"]["reasons"])} for v, s, p in rows])
    write_once(os.path.join(out, "summary.md"),
               f"# Stage 1 — plan `{plan['name']}`, run `{run_id}`\n\nValidation fingerprint "
               f"`{manifest['validation_fingerprint']}`; git `{manifest['git']['commit']}` "
               f"(dirty={manifest['git']['dirty']}).\n\nVAL {val[0]}..{val[1]}. Only eligible candidates may enter "
               f"TEST.\n\n{md_table(summary)}\n")
    return run_id


def _train_ml_on(data, bt, strat, dev):
    from core.ml_filter import train_filter
    frames = {p: V.segment(df, dev[0], dev[1], history=0) for p, df in data.items()}
    frames = {p: f for p, f in frames.items() if len(f) > 1000}
    return train_filter(frames, bt.exit_rules, horizon=48, threshold=bt.ml_threshold, strategy=strat,
                        signal_threshold=bt.threshold, save_path=None, fee_rate=bt.fee_rate,
                        stop_slippage_pct=bt.stop_slippage_pct)


# ------------------------------------------------------------------ stage 2
def stage2(plan_path: str, run_id: str, allow_dirty: bool = False, argv=None) -> dict:
    plan = load_plan(plan_path)
    store = store_for(plan)
    s1 = store.read_json(plan["name"], run_id, "stage1", "provenance.json")
    decisions = store.read_json(plan["name"], run_id, "stage1", "eligibility.json")
    if os.path.isdir(store.stage_dir(plan["name"], run_id, "stage2")):
        raise ImmutableError(f"stage2 already exists for run {run_id}")
    git = git_state()
    if git["dirty"] and not allow_dirty:
        raise RefusedError("working tree has uncommitted changes; commit first")
    bt = base_config(plan)
    data_prov, comps = current_components(plan, bt)
    if validation_fingerprint(comps) != s1["validation_fingerprint"]:
        changed = diff_components(s1["fingerprint_components"], comps)
        raise RefusedError("validation inputs changed since Stage 1 (" + ", ".join(changed[:10]) +
                           "). This needs a NEW validation version: run Stage 1 again (it creates a new run).")
    test = plan["periods"]["TEST"]
    store.check_test_window_unused(*test, plan["symbols"], plan["interval"])
    eligible = {k: d for k, d in decisions.items() if d.get("eligible")}
    out = store.stage_dir(plan["name"], run_id, "stage2", create=True)
    results = {}
    if not eligible:
        results = {"evaluated": [], "note": "No candidate passed the validation eligibility rules. "
                                            "TEST was NOT evaluated and the TEST window was NOT consumed."}
    else:
        data = load_data(plan)
        rows = []
        for name, d in eligible.items():
            bundle = None
            if d["variant"] == "fixed+ML":
                path = os.path.join(store.run_dir(plan["name"], run_id), "stage1", f"ml_{d['strategy']}.pkl")
                if sha256_file(path) != s1["models"].get(d["strategy"]):
                    raise RefusedError(f"model file for {d['strategy']} changed since Stage 1")
                bundle = joblib.load(path)
            p = V.pooled(V.run_period(data, bt, d["strategy"], d.get("params"), *test, ml_bundle=bundle))
            ok, why = V.criteria(p)
            rows.append({"candidate": name, "trades": p["trades"], "expectancy_r": p["expectancy_r"],
                         "ci95_low_r": p["ci_low"], "profit_factor": p["profit_factor"], "breadth": p["breadth"],
                         "worst_dd_pct": p["worst_max_dd_pct"], "credible_edge": ok,
                         "reasons": "; ".join(why) or "passed VAL eligibility and all TEST criteria"})
        results = {"evaluated": rows}
        store.register_consumed(f"{plan['name']}-{run_id}", *test, plan["symbols"], plan["interval"],
                                note=f"final test of run {run_id}")
    manifest = build_manifest(run_id, "stage2", resolve(plan["protocol"]), plan, data_prov, bt.as_dict(),
                              {k: STRATEGIES[k] for k in STRATEGIES}, s1.get("models"),
                              extra={"stage1_fingerprint": s1["validation_fingerprint"], "allow_dirty": allow_dirty},
                              argv=argv)
    write_once(os.path.join(out, "provenance.json"), manifest)
    write_once(os.path.join(out, "result.json"), results)
    return results


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["stage1", "stage2", "list"])
    ap.add_argument("--plan", required=True)
    ap.add_argument("--run-id")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--supersedes")
    ap.add_argument("--reason", default="")
    a = ap.parse_args(argv)
    try:
        if a.command == "stage1":
            print("created run", stage1(a.plan, a.allow_dirty, a.supersedes, a.reason, argv=sys.argv))
        elif a.command == "stage2":
            if not a.run_id:
                raise RefusedError("--run-id is required")
            print(json.dumps(stage2(a.plan, a.run_id, a.allow_dirty, argv=sys.argv), indent=1, default=str))
        else:
            plan = load_plan(a.plan)
            for r in store_for(plan).list_runs(plan["name"]):
                print(r)
    except (RefusedError, ImmutableError) as exc:
        print(f"REFUSED: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
