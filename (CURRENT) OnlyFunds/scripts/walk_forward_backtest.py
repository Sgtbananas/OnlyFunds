"""Walk-forward validation: optimise on each training window, test on the next unseen window.

    python scripts/walk_forward_backtest.py [--pairs BTCUSDT] [--interval 15m] [--folds 4]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.autotune import walk_forward  # noqa: E402
from core.core_data import fetch_klines  # noqa: E402
from utils.config import APP_ROOT, build_settings, load_config  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+")
    ap.add_argument("--interval")
    ap.add_argument("--profile")
    ap.add_argument("--folds", type=int, default=4)
    a = ap.parse_args(argv)
    cfg = load_config()
    bt, _ = build_settings(cfg, a.profile, {"interval": a.interval} if a.interval else None)
    out = {}
    for pair in a.pairs or cfg.trading_pairs:
        try:
            df = fetch_klines(pair, bt.interval, 1000)
            r = walk_forward(df, bt, n_folds=a.folds, pair=pair)
        except Exception as exc:
            print(f"{pair}: skipped ({exc})")
            continue
        print(f"\n{pair}: compounded OOS return {r['oos_compounded_return_pct']:.2f}% "
              f"({r['profitable_folds']}/{r['n_folds']} folds profitable)")
        print(r["folds"].to_string(index=False))
        out[pair] = {"oos_compounded_return_pct": r["oos_compounded_return_pct"],
                     "folds": r["folds"].to_dict("records")}
    path = os.path.join(APP_ROOT, "state", "walk_forward.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved {path}")
    return 0 if out else 1


if __name__ == "__main__":
    sys.exit(main())
