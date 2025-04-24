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
    Dynamically tune threshold by scanning from 0.1 to 0.9 in 0.05 steps
    and choosing the threshold which gave the highest average backtest return.
    """
    try:
        signal = smooth_signal(generate_signal(df))
        prices = df["Close"]

        best_thresh = 0.1
        best_return = -np.inf

        # Scan for the best threshold
        for t in np.arange(0.1, 1.0, 0.05):
            bt = run_backtest(signal, prices, threshold=t)
            if bt.empty:
                continue
            avg_ret = bt["return"].mean()
            logging.debug(f"Threshold {t:.2f} → avg_return {avg_ret:.4f}")
            if avg_ret > best_return:
                best_return = avg_ret
                best_thresh = t

        logging.info(f"✨ Chosen threshold = {best_thresh:.2f} (avg_return={best_return:.4f})")
        return round(best_thresh, 2)

    except Exception as e:
        logging.error(f"adaptive_threshold failed: {e}")
        return 0.5

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
