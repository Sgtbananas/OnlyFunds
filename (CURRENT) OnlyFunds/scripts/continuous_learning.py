"""Retrain the ML entry filter on the configured pairs and print its validation report.

    python scripts/continuous_learning.py [--pairs BTCUSDT ETHUSDT SOLUSDT] [--interval 15m]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.core_data import fetch_klines  # noqa: E402
from core.ml_filter import MODEL_PATH, train_filter  # noqa: E402
from utils.config import build_settings, load_config  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+")
    ap.add_argument("--interval")
    a = ap.parse_args(argv)
    cfg = load_config()
    bt, _ = build_settings(cfg, None, {"interval": a.interval} if a.interval else None)
    frames = {}
    for pair in a.pairs or cfg.trading_pairs:
        try:
            frames[pair] = fetch_klines(pair, bt.interval, 1000)
        except Exception as exc:
            print(f"{pair}: skipped ({exc})")
    if not frames:
        print("No data available.")
        return 1
    bundle = train_filter(frames, bt.exit_rules, horizon=int(cfg.section("ml").get("horizon", 48)),
                          threshold=bt.ml_threshold, strategy=bt.strategy, signal_threshold=bt.threshold,
                          save_path=MODEL_PATH)
    print(json.dumps(bundle["report"], indent=2, default=str))
    print("Model shows an out-of-sample edge." if bundle["report"]["useful"]
          else "Model shows NO out-of-sample edge; keep ml.enabled: false.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
