"""Evaluate every strategy (and the ML filter) for a credible edge.

    python scripts/evaluate_strategies.py --csv path/to/BTCUSDT_15m.csv [more.csv ...] --interval 15m
    python scripts/evaluate_strategies.py --cache            # every file in data/cache/
    python scripts/evaluate_strategies.py --synthetic        # controlled regimes (logic check, NOT market evidence)
    add --out report.md to save a markdown report

Reports net-of-cost metrics, walk-forward out-of-sample results, ML on/off
comparison and in-sample vs out-of-sample degradation of tuned parameters.
"""
import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd  # noqa: E402

from core.autotune import ml_out_of_sample_backtest, optimise  # noqa: E402
from core.core_data import cache_dir, normalize_ohlcv  # noqa: E402
from core.evaluation import evaluate  # noqa: E402
from utils.config import build_settings, load_config  # noqa: E402


def load_frames(a):
    frames = {}
    if a.synthetic:
        from tests.synthetic import synthetic_regime
        for k in ("trend_up", "trend_down", "range", "high_vol", "random_walk", "switching"):
            frames[f"synthetic:{k}"] = synthetic_regime(k, a.bars, seed=7, interval=a.interval)
    paths = list(a.csv or [])
    if a.cache:
        paths += sorted(glob.glob(os.path.join(cache_dir(), f"*_{a.interval}.csv")))
    for p in paths:
        frames[os.path.splitext(os.path.basename(p))[0]] = normalize_ohlcv(pd.read_csv(p))
    return frames


def md(df: pd.DataFrame) -> str:
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: f"{v:.3f}")
    head = "| " + " | ".join(d.columns) + " |\n|" + "---|" * len(d.columns) + "\n"
    return head + "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in d.itertuples(index=False))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+")
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--bars", type=int, default=4000)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--no-ml", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    frames = load_frames(a)
    if not frames:
        print("No data. Use --csv, --cache or --synthetic.")
        return 1
    bt, _ = build_settings(load_config(), a.profile, {"interval": a.interval})
    t0 = time.time()
    res = evaluate(frames, bt, walk_forward_folds=a.folds)
    ml_rows, deg_rows = [], []
    for name, df in frames.items():
        if not a.no_ml:
            try:
                r = ml_out_of_sample_backtest(df, bt, name)
                rep = r["report"]
                ml_rows.append({"dataset": name, "useful": rep["useful"], "val_auc": rep["validation_auc"],
                                "exp_r_all": rep["validation_expectancy_r_all"],
                                "exp_r_accepted": rep["validation_expectancy_r_accepted"],
                                "rule_return_pct": r["rule_only"]["total_return_pct"],
                                "ml_return_pct": r["ml_gated"]["total_return_pct"],
                                "rule_pf": r["rule_only"]["profit_factor"], "ml_pf": r["ml_gated"]["profit_factor"],
                                "rule_trades": r["rule_only"]["trades"], "ml_trades": r["ml_gated"]["trades"]})
            except ValueError as exc:
                ml_rows.append({"dataset": name, "useful": False, "note": str(exc)[:80]})
        o = optimise(df, bt, pair=name)
        if o["best_params"] is not None:
            deg_rows.append({"dataset": name, "best_params": str(o["best_params"]),
                             "in_sample_return_pct": o["in_sample"]["total_return_pct"],
                             "oos_return_pct": o["out_of_sample"]["total_return_pct"],
                             "oos_buy_hold_pct": o["out_of_sample"]["buy_and_hold_return_pct"],
                             "confirmed": o["confirmed"]})
    sections = [
        ("Fixed default parameters (full period, after fees and slippage)", res["fixed"]),
        ("Walk-forward (tuned on each training window, scored on the next unseen window)", res["walk_forward"]),
        ("Verdict", res["verdict"]),
        ("ML filter: rule-only vs ML-gated on the unseen last 30%", pd.DataFrame(ml_rows)),
        ("Overfitting check: tuned on first 70%, scored on last 30%", pd.DataFrame(deg_rows)),
    ]
    text = f"# Strategy evaluation\n\nDatasets: {', '.join(frames)}; interval {a.interval}; " \
           f"profile {a.profile or 'config default'}; runtime {time.time() - t0:.0f}s.\n"
    if a.synthetic:
        text += ("\n> Synthetic regimes test that the logic behaves as designed. "
                 "They are **not** evidence about real markets.\n")
    for title, table in sections:
        text += f"\n## {title}\n\n" + (md(table) if len(table) else "_no rows_") + "\n"
    print(text)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
