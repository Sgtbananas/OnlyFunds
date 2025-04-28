import os
import pandas as pd
import numpy as np
from datetime import timedelta
from utils.helpers import save_json
from core.backtester import run_backtest
from core.core_signals import smooth_signal, generate_signal
import optuna

TRAIN_DAYS = 30
PARAM_SPACE = {
    "threshold": (0.05, 0.99)
}

def load_klines(pair, interval, lookback):
    from core.core_data import fetch_klines
    df = fetch_klines(pair, interval, lookback)
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return df

def objective(trial, df_train):
    threshold = trial.suggest_float("threshold", *PARAM_SPACE["threshold"])
    signal = smooth_signal(generate_signal(df_train))
    res = run_backtest(signal, df_train["Close"], threshold=threshold)
    if "type" in res.columns and "total_return" in res.columns:
        total_return = res[res["type"] == "summary"]["total_return"].iloc[0]
    else:
        total_return = 0
    return total_return

def walkforward_optimize(pair, interval="1h"):
    df = load_klines(pair, interval, lookback=None)
    results = {}
    min_date = df.index.min().normalize() + timedelta(days=TRAIN_DAYS)
    max_date = df.index.max().normalize() - timedelta(days=1)
    for day in pd.date_range(min_date, max_date):
        train_end = day
        train_start = train_end - timedelta(days=TRAIN_DAYS)
        test_day = train_end + timedelta(days=1)
        df_train = df[(df.index >= train_start) & (df.index < train_end)]
        df_test = df[(df.index >= test_day) & (df.index < test_day + timedelta(days=1))]
        if len(df_train) < TRAIN_DAYS*12:
            continue
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda t: objective(t, df_train), n_trials=30, show_progress_bar=False)
        best_params = study.best_params
        signal = smooth_signal(generate_signal(df_test))
        res = run_backtest(signal, df_test["Close"], threshold=best_params["threshold"])
        results[str(test_day.date())] = {
            "pair": pair,
            "interval": interval,
            "threshold": best_params["threshold"],
            "test_day": str(test_day.date()),
            "test_return": float(res[res["type"] == "summary"]["total_return"].iloc[0]) if "type" in res.columns and "total_return" in res.columns and not res.empty else 0
        }
    return results

def main():
    from core.core_data import TRADING_PAIRS
    all_params = {}
    for pair in TRADING_PAIRS:
        print(f"Optimizing {pair} walk-forward…")
        per_day_params = walkforward_optimize(pair)
        all_params[pair] = per_day_params
    save_json(all_params, "state/auto_params.json", indent=2)
    print("Saved walk-forward optimized params to state/auto_params.json")

if __name__ == "__main__":
    main()
