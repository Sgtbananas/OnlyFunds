#!/usr/bin/env python3
import pandas as pd, numpy as np, json
from datetime import datetime, timedelta
from core.core_data    import fetch_klines, add_indicators, TRADING_PAIRS
from core.core_signals import generate_signal
from core.backtester   import run_backtest
from joblib import Parallel, delayed

RESULTS = []

# Window params
window_len = 1000
test_len = 200
max_data = 1500

def walk_window(start):
    window_result = {"window_start": start, "pairs": []}
    for pair in TRADING_PAIRS:
        # Fetch and slice data for this pair
        df = fetch_klines(pair=pair, interval="15m", limit=max_data)
        if df.empty or len(df) < (start+window_len+test_len):
            continue
        df = add_indicators(df)
        train = df.iloc[start : start+window_len]
        test = df.iloc[start+window_len : start+window_len+test_len]
        # You could train your meta-learner here on "train"
        sig = generate_signal(test)
        test_summary = run_backtest(
            signal=sig,
            prices=test["Close"],
            threshold=0.05,
            initial_capital=10.0,
            risk_pct=0.01,
            stop_loss_pct=0.005,
            take_profit_pct=0.01,
        )
        summary_row = test_summary.loc[test_summary.type=="summary"]
        if not summary_row.empty:
            summary_dict = summary_row.iloc[0].to_dict()
            summary_dict["pair"] = pair
            window_result["pairs"].append(summary_dict)
    return window_result

# Parallelize window sweeping!
starts = list(range(0, max_data-window_len-test_len+1, test_len))
window_results = Parallel(n_jobs=-1, verbose=10)(
    delayed(walk_window)(start) for start in starts
)

# Collate global results
RESULTS = window_results

with open("state/walk_forward.json","w") as f:
    json.dump(RESULTS, f, indent=2)
print("✅ Walk-forward validation complete → state/walk_forward.json")