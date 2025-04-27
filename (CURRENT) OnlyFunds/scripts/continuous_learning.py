#!/usr/bin/env python3
import json, joblib
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from core.core_data    import fetch_klines, add_indicators, TRADING_PAIRS
from core.core_signals import generate_signal
from core.backtester   import run_backtest
from utils.helpers     import load_json, save_json

STATE_DIR = Path("state")
MODEL_PATH = STATE_DIR/"meta_model.pkl"
LOG_PATH   = STATE_DIR/"trade_log.json"

def build_dataset(window=100):
    # 1) Fetch history for each pair
    rows = []
    for pair in TRADING_PAIRS:
        df = fetch_klines(pair, interval="15m", limit=window+1)
        df = add_indicators(df)
        sig = generate_signal(df)
        # 2) Backtest with default params for each mode
        #    (you could pre-calc these or fetch from state/backtest_results.json)
        summary = run_backtest(sig, df["Close"], threshold=0.05, initial_capital=10)
        sharpe = summary.loc[summary.type=="summary", "sharpe_ratio"].values[0]
        vol    = df["Close"].pct_change().rolling(14).std().iloc[-1]
        regime = df["regime"].iloc[-1]   # assuming you added HMM tags as 'regime'
        rows.append({
            "pair": pair,
            "sharpe": sharpe,
            "volatility": vol,
            "regime": regime,
            # label = best-performing mode in last window
            "label": summary.loc[summary.type=="summary", "best_mode"].values[0]
        })
    return pd.DataFrame(rows)

def retrain():
    print(f"[{datetime.utcnow()}] Building dataset…")
    data = build_dataset()
    X = pd.get_dummies(data[["sharpe","volatility","regime"]])
    y = data["label"]
    print(f"[{datetime.utcnow()}] Training meta-learner…")
    model = LogisticRegression(multi_class="multinomial", max_iter=500)
    model.fit(X, y)
    STATE_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    save_json({"last_retrain": datetime.utcnow().isoformat()}, STATE_DIR/"learning_meta.json")
    print(f"[{datetime.utcnow()}] Saved model → {MODEL_PATH}")

if __name__ == "__main__":
    retrain()
