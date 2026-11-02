"""Backtest every configured pair and write state/backtest_results.json.

    python scripts/backtest_suite.py [--pairs BTCUSDT ETHUSDT] [--interval 15m] [--profile Normal] [--oos]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd  # noqa: E402

from core.backtester import run_backtest, run_out_of_sample  # noqa: E402
from core.core_data import fetch_klines  # noqa: E402
from utils.config import APP_ROOT, build_settings, load_config  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+")
    ap.add_argument("--interval")
    ap.add_argument("--profile")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--oos", action="store_true", help="report in-sample vs out-of-sample (70/30)")
    a = ap.parse_args(argv)
    cfg = load_config()
    bt, _ = build_settings(cfg, a.profile, {"interval": a.interval} if a.interval else None)
    rows, results = [], []
    for pair in a.pairs or cfg.trading_pairs:
        try:
            df = fetch_klines(pair, bt.interval, a.limit)
        except Exception as exc:
            print(f"{pair}: no data ({exc})")
            continue
        if a.oos:
            for seg, res in run_out_of_sample(df, bt, pair).items():
                rows.append({"pair": pair, "segment": seg, **res.metrics})
        else:
            res = run_backtest(df, bt, pair)
            rows.append({"pair": pair, "segment": "full", **res.metrics})
            results.append({"pair": pair, "metrics": res.metrics, "trades": res.trades.to_dict("records")})
    if not rows:
        print("No results (no data could be loaded).")
        return 1
    table = pd.DataFrame(rows).drop(columns=["exit_reasons"])
    cols = ["pair", "segment", "trades", "total_return_pct", "buy_and_hold_return_pct", "win_rate_pct",
            "max_drawdown_pct", "profit_factor", "fees_paid"]
    print(table[cols].round(3).to_string(index=False))
    out = os.path.join(APP_ROOT, "state", "backtest_results.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"time": time.time(), "config": bt.as_dict(), "results": results or rows}, f, default=str)
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
