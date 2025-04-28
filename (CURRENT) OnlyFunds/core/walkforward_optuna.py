import os
import pandas as pd
import numpy as np
from datetime import timedelta
from utils.helpers import save_json
from core.backtester import run_backtest
from core.core_signals import smooth_signal, generate_signal
from core.core_data import fetch_klines, add_indicators
import optuna

TRAIN_DAYS = 30
OPTUNA_TRIALS = 30

# Parameter search spaces for indicator params and threshold
PARAM_SPACE = {
    "threshold": (0.05, 0.99),
    "regime_vol_window": (10, 40),      # int
    "regime_mom_window": (10, 40),      # int
    "regime_vol_thresh": (0.005, 0.03), # float
    "regime_mom_thresh": (0.002, 0.02), # float
    "trend_ema_fast_window": (10, 40),  # int
    "trend_ema_slow_window": (20, 100), # int
    "reversion_rsi_low": (10, 40),      # int
    "reversion_rsi_high": (60, 90),     # int
    "rsi_window": (8, 28),              # int
    "macd_fast": (8, 18),               # int
    "macd_slow": (18, 40),              # int
    "macd_signal": (6, 15),             # int
    "boll_window": (10, 40),            # int
    "boll_std": (1.5, 3.0),             # float
    "ema_span": (10, 40),               # int
    "volatility_window": (5, 25),       # int
    "smoothing_window": (2, 10),        # int
}

def load_klines(pair, interval, lookback):
    df = fetch_klines(pair, interval, lookback)
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return df

def get_indicator_params(trial):
    # Assemble all indicator params for add_indicators and signal functions
    indicator_params = {
        "rsi_window":         trial.suggest_int("rsi_window", *PARAM_SPACE["rsi_window"]),
        "macd_fast":          trial.suggest_int("macd_fast", *PARAM_SPACE["macd_fast"]),
        "macd_slow":          trial.suggest_int("macd_slow", *PARAM_SPACE["macd_slow"]),
        "macd_signal":        trial.suggest_int("macd_signal", *PARAM_SPACE["macd_signal"]),
        "boll_window":        trial.suggest_int("boll_window", *PARAM_SPACE["boll_window"]),
        "boll_std":           trial.suggest_float("boll_std", *PARAM_SPACE["boll_std"]),
        "ema_span":           trial.suggest_int("ema_span", *PARAM_SPACE["ema_span"]),
        "volatility_window":  trial.suggest_int("volatility_window", *PARAM_SPACE["volatility_window"]),
    }
    signal_params = {
        "regime_kwargs": {
            "vol_window":    trial.suggest_int("regime_vol_window", *PARAM_SPACE["regime_vol_window"]),
            "mom_window":    trial.suggest_int("regime_mom_window", *PARAM_SPACE["regime_mom_window"]),
            "vol_thresh":    trial.suggest_float("regime_vol_thresh", *PARAM_SPACE["regime_vol_thresh"]),
            "mom_thresh":    trial.suggest_float("regime_mom_thresh", *PARAM_SPACE["regime_mom_thresh"]),
        },
        "trend_kwargs": {
            "ema_fast_window": trial.suggest_int("trend_ema_fast_window", *PARAM_SPACE["trend_ema_fast_window"]),
            "ema_slow_window": trial.suggest_int("trend_ema_slow_window", *PARAM_SPACE["trend_ema_slow_window"]),
        },
        "reversion_kwargs": {
            "rsi_low":  trial.suggest_int("reversion_rsi_low", *PARAM_SPACE["reversion_rsi_low"]),
            "rsi_high": trial.suggest_int("reversion_rsi_high", *PARAM_SPACE["reversion_rsi_high"]),
        },
    }
    smoothing_window = trial.suggest_int("smoothing_window", *PARAM_SPACE["smoothing_window"])
    threshold = trial.suggest_float("threshold", *PARAM_SPACE["threshold"])
    return indicator_params, signal_params, smoothing_window, threshold

def objective(trial, df_train):
    indicator_params, signal_params, smoothing_window, threshold = get_indicator_params(trial)
    df_ind = add_indicators(df_train, indicator_params)
    if df_ind.empty:
        return -1e9  # fail fast on bad indicator config
    signal = smooth_signal(generate_signal(df_ind, indicator_params=signal_params), smoothing_window=smoothing_window)
    res = run_backtest(signal, df_ind["Close"], threshold=threshold)
    if "type" in res.columns and "total_pnl" in res.columns:
        total_pnl = res[res["type"] == "summary"]["total_pnl"].iloc[0]
    else:
        total_pnl = 0
    return total_pnl

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
        if len(df_train) < TRAIN_DAYS * 12 or df_test.empty:
            continue
        def obj_func(trial):
            return objective(trial, df_train)
        study = optuna.create_study(direction="maximize")
        study.optimize(obj_func, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
        best_params = study.best_params
        # --- Out-of-sample test
        indicator_params, signal_params, smoothing_window, threshold = get_indicator_params(study.best_trial)
        df_test_ind = add_indicators(df_test, indicator_params)
        if df_test_ind.empty:
            test_return = 0
        else:
            signal = smooth_signal(generate_signal(df_test_ind, indicator_params=signal_params), smoothing_window=smoothing_window)
            res = run_backtest(signal, df_test_ind["Close"], threshold=threshold)
            if "type" in res.columns and "total_pnl" in res.columns and not res.empty:
                test_return = float(res[res["type"] == "summary"]["total_pnl"].iloc[0])
            else:
                test_return = 0
        results[str(test_day.date())] = {
            "pair": pair,
            "interval": interval,
            "params": best_params,
            "test_day": str(test_day.date()),
            "test_return": test_return,
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