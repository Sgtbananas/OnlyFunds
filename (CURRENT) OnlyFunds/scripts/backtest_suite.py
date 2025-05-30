"""
Parallelized grid search backtest suite for OnlyFunds.
- Sweeps all pairs and all param grid combos.
- Uses joblib for parallelism.
- Aggregates results into state/backtest_results.json for analysis.
"""
import os
import itertools
import json
from joblib import Parallel, delayed
from core.core_data import fetch_klines, add_indicators, TRADING_PAIRS
from core.core_signals import generate_signal
from core.backtester import run_backtest

# 1) Pairs to test
pairs = TRADING_PAIRS

# 2) Grid of hyperparameters to explore (edit as desired)
param_grid = {
    "threshold":      [0.01, 0.03, 0.05, 0.1],
    "risk_pct":       [0.005, 0.01, 0.02],
    "stop_loss_pct":  [0.003, 0.005, 0.01],
    "take_profit_pct":[0.005, 0.01, 0.02],
}

# 3) Make every combination: (pair, threshold, risk_pct, ...)
combos = list(itertools.product(
    pairs,
    param_grid["threshold"],
    param_grid["risk_pct"],
    param_grid["stop_loss_pct"],
    param_grid["take_profit_pct"],
))

def backtest_one(pair, threshold, risk_pct, stop_loss_pct, take_profit_pct):
    try:
        # fetch & prep data
        df = fetch_klines(pair=pair, interval="15m", limit=1000)
        if df.empty:
            return {
                "pair": pair,
                "error": "No data fetched",
                "threshold": threshold,
                "risk_pct": risk_pct,
                "stop_loss_pct": stop_loss_pct,
                "take_profit_pct": take_profit_pct,
            }
        df = add_indicators(df)
        sig = generate_signal(df)
        # run backtest
        df_combined = run_backtest(
            signal=sig,
            prices=df["Close"],
            threshold=threshold,
            initial_capital=10.0,
            risk_pct=risk_pct,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
        # extract summary row
        summary = df_combined.loc[df_combined["type"]=="summary"].iloc[0].to_dict()
        # annotate with params
        summary.update({
            "pair": pair,
            "threshold": threshold,
            "risk_pct": risk_pct,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
        })
        return summary
    except Exception as e:
        return {
            "pair": pair,
            "error": str(e),
            "threshold": threshold,
            "risk_pct": risk_pct,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
        }

if __name__ == "__main__":
    os.makedirs("state", exist_ok=True)
    # run in parallel on all CPUs
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(backtest_one)(*cfg) for cfg in combos
    )
    # write out to JSON
    with open("state/backtest_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Completed {len(results)} backtests → state/backtest_results.json")