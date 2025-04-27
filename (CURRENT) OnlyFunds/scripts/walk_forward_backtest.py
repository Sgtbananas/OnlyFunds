#!/usr/bin/env python3
import pandas as pd, numpy as np, json
from datetime import datetime, timedelta
from core.core_data    import fetch_klines, add_indicators, TRADING_PAIRS
from core.core_signals import generate_signal
from core.backtester   import run_backtest

RESULTS = []

# rolling windows of 1000 bars, step 200 bars
for start in range(0, 1500, 200):
    train_df = {}  # build your meta-learner on data[start : start+1000]
    test_df  = {}  # then backtest selector on data[start+1000 : start+1000+200]
    # … similar to above, but ensure no look-ahead …
    # After each slice, record:
    RESULTS.append({
      "window_start": start,
      "total_return": test_summary.loc[test_summary.type=="summary","total_return"].values[0],
      "sharpe":       test_summary.loc[test_summary.type=="summary","sharpe_ratio"].values[0],
    })

# Write out
with open("state/walk_forward.json","w") as f:
    json.dump(RESULTS, f, indent=2)
print("✅ Walk-forward validation complete → state/walk_forward.json")
