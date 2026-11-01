"""Optimise parameters per pair (train 70% → test 30%) and save the report.

Results go to state/auto_params.json with a ``confirmed`` flag. Nothing is
applied automatically: copy confirmed values into config.yaml yourself.

    python scripts/auto_tuner.py [--pairs BTCUSDT ETHUSDT] [--interval 15m]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.autotune import optimise  # noqa: E402
from core.core_data import fetch_klines  # noqa: E402
from utils.config import APP_ROOT, build_settings, load_config  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+")
    ap.add_argument("--interval")
    ap.add_argument("--profile")
    a = ap.parse_args(argv)
    cfg = load_config()
    bt, _ = build_settings(cfg, a.profile, {"interval": a.interval} if a.interval else None)
    report = {}
    for pair in a.pairs or cfg.trading_pairs:
        try:
            r = optimise(fetch_klines(pair, bt.interval, 1000), bt, pair=pair)
        except Exception as exc:
            print(f"{pair}: skipped ({exc})")
            continue
        report[pair] = {"best_params": r["best_params"], "confirmed": r["confirmed"], "reason": r["reason"],
                        "in_sample": r.get("in_sample"), "out_of_sample": r.get("out_of_sample")}
        print(f"{pair}: {r['best_params']} -> {'CONFIRMED' if r['confirmed'] else 'not confirmed'} ({r['reason']})")
    path = os.path.join(APP_ROOT, "state", "auto_params.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Saved {path}")
    return 0 if report else 1


if __name__ == "__main__":
    sys.exit(main())
