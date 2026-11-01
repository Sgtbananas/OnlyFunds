"""Fill the local candle cache for the configured pairs (run periodically to build longer history)."""
import sys

from data.historical.historical import save_historical_data
from utils.config import load_config

if __name__ == "__main__":
    cfg = load_config()
    interval = sys.argv[1] if len(sys.argv) > 1 else cfg.default_interval
    for pair in cfg.trading_pairs:
        try:
            save_historical_data(pair, interval, 1000)
        except Exception as exc:
            print(f"{pair}: failed ({exc})")
    print("Historical data update complete.")
