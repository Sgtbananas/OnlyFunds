# core/core_signals.py

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from core.backtester import run_backtest  # for adaptive thresholding

load_dotenv()
LOG_FILE = os.getenv("SIGNAL_LOG", "signals_log.json")

def generate_signal(df: pd.DataFrame) -> pd.Series:
    """
    Composite signal: RSI, MACD diff, EMA diff, BB position.
    Clipped to [-1,1].
    """
    try:
        sig = (
            0.4 * (df["rsi"] - 50) / 50
          + 0.3 * df["macd"]
          + 0.2 * df["ema_diff"] / df["Close"]
          + 0.1 * ((df["Close"] - df["bollinger_mid"]) /
                   (df["bollinger_upper"] - df["bollinger_lower"]))
        )
        return sig.clip(-1, 1)
    except Exception as e:
        logging.error(f"generate_signal error: {e}")
        return pd.Series(0, index=df.index)

def smooth_signal(signal: pd.Series, window: int = 5) -> pd.Series:
    """Rolling mean to smooth noise."""
    return signal.rolling(window, min_periods=1).mean()

def adaptive_threshold(df: pd.DataFrame, target_profit: float = 0.01) -> float:
    """
    Grid-search threshold ∈ [0.1,0.95] maximizing avg return.
    """
    best_t, best_ret = 0.5, -np.inf
    sig = smooth_signal(generate_signal(df))
    for t in np.arange(0.1, 1.0, 0.05):
        bt = run_backtest(sig, df["Close"], threshold=t)
        avg = bt["return"].mean() if "return" in bt.columns else -np.inf
        if avg > best_ret:
            best_ret, best_t = avg, t
    return round(best_t, 2)

def track_trade_result(resp: dict, pair: str, action: str):
    """Append a JSON record to disk."""
    rec = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "pair":      pair,
        "action":    action,
        "response":  resp
    }
    try:
        data = []
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r") as f:
                data = json.load(f)
        data.append(rec)
        with open(LOG_FILE, "w") as f:
            json.dump(data, f, indent=2)
        logging.info(f"Logged trade: {action} {pair}")
    except Exception as e:
        logging.error(f"track_trade_result error: {e}")
