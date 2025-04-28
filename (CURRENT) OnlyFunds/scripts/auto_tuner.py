import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import optuna
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from core.core_data import fetch_klines, validate_df, add_indicators, TRADING_PAIRS
from core.core_signals import smooth_signal
from core.backtester import run_backtest
from utils.helpers import load_json, save_json
from core.ml_filter import (
    load_trade_log, extract_features_labels, train_and_save_model,
    load_model, ml_confidence, fit_best_ml_model, extract_feature_array
)
import warnings

warnings.filterwarnings("ignore")

STATE_DIR = "state"
AUTO_PARAMS_FILE = os.path.join(STATE_DIR, "auto_params.json")
MODEL_PATH = os.path.join(STATE_DIR, "ml_model.pkl")
TRADE_LOG_FILE = os.path.join(STATE_DIR, "trade_log.json")
CONFIG_FILE = "config.json"
if os.path.exists(CONFIG_FILE):
    TRADING_CFG = load_json(CONFIG_FILE)["trading"]
else:
    TRADING_CFG = {}
DEFAULT_THRESHOLD = TRADING_CFG.get("threshold", 0.1)
DEFAULT_LOOKBACK = 1000

interval_grid = ["5m", "15m", "30m", "1h", "4h", "1d"]
lookback_grid = [400, 600, 800, 1000, 1200, 1600]
threshold_grid = np.round(np.arange(0.05, 0.6, 0.05), 3).tolist()

def get_price_df(pair, interval, lookback):
    df = fetch_klines(pair, interval, lookback)
    if not df.empty and validate_df(df):
        df = add_indicators(df)
    return df

def get_best_model():
    # Retrain on all trade logs, using the best available estimator
    df = load_trade_log()
    if df.empty:
        return None
    X, y = extract_features_labels(df)
    model, msg = fit_best_ml_model(X, y, MODEL_PATH)
    print(msg)
    return model

def objective(trial, pair, model):
    interval = trial.suggest_categorical("interval", interval_grid)
    lookback = trial.suggest_categorical("lookback", lookback_grid)
    threshold = trial.suggest_categorical("threshold", threshold_grid)
    df = get_price_df(pair, interval, lookback)
    if df.empty or not validate_df(df):
        trial.set_user_attr("skip", True)
        return -999
    # ML ensemble signal
    try:
        feats = []
        for i in range(len(df)):
            feats.append(extract_feature_array(df.iloc[i]))
        feats = np.array(feats)
        if len(feats) == 0: return -999
        pred_probs = model.predict_proba(feats)[:,1] if hasattr(model, "predict_proba") else model.predict(feats)
        sig = pd.Series(pred_probs, index=df.index)
    except Exception as e:
        print(f"[{pair}] ML signal error: {e}")
        return -999
    prices = df["Close"]
    bt = run_backtest(
        sig,
        prices,
        threshold=threshold,
        initial_capital=TRADING_CFG.get("default_capital", 1000),
        risk_pct=TRADING_CFG.get("per_trade", 0.01),
        stop_loss_pct=TRADING_CFG.get("stop_loss_pct", 0.01),
        take_profit_pct=TRADING_CFG.get("take_profit_pct", 0.02),
        fee_pct=TRADING_CFG.get("fee", 0.0004),
    )
    summary = bt.iloc[0] if "type" in bt.columns and bt.iloc[0].get("type") == "summary" else None
    if summary is not None and summary.get("trades", 0) > 3:
        return summary.get("sharpe_ratio", 0)
    return -999

def tune_pair(pair, model, n_trials=40, timeout=120):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, pair, model), n_trials=n_trials, timeout=timeout)
    best = study.best_params
    best_value = study.best_value
    print(f"[{pair}] Best: {best} (Sharpe={best_value:.3f})")
    return pair, {**best, "sharpe": best_value}

def tune_all_pairs(pairs=TRADING_PAIRS, model=None, n_trials=40, timeout=120):
    results = {}
    with ThreadPoolExecutor(max_workers=min(4, len(pairs))) as pool:
        futs = [pool.submit(tune_pair, pair, model, n_trials, timeout) for pair in pairs]
        for fut in tqdm(futs):
            try:
                pair, best = fut.result()
                results[pair] = best
            except Exception as e:
                print(f"[ERROR] {e}")
    return results

def main():
    print("=== Rolling Optuna Auto-Tuner w/ ML Ensemble ===")
    print(f"Tuning pairs: {TRADING_PAIRS}")
    print("[INFO] Training the ML ensemble model on trade log ...")
    model = get_best_model()
    if model is None:
        print("[WARN] No trade or not enough data to train ML model. Aborting tuner.")
        return
    auto_params = tune_all_pairs(TRADING_PAIRS, model=model, n_trials=40, timeout=180)
    sharpe_scores = [v.get("sharpe", -999) for v in auto_params.values()]
    global_best = max(auto_params.values(), key=lambda v: v.get("sharpe", -999), default=None)
    if global_best:
        auto_params["global"] = {k: v for k, v in global_best.items() if k in ["interval","lookback","threshold"]}
    save_json(auto_params, AUTO_PARAMS_FILE, indent=2)
    print(f"\nSaved best params to {AUTO_PARAMS_FILE}")
    print(auto_params)
    print(f"\nML model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()