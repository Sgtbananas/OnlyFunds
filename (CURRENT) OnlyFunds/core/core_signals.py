# core/core_signals.py

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# backtester is assumed to expose run_backtest(signals: pd.Series, prices: pd.Series, threshold: float) -> pd.DataFrame
from core.backtester import run_backtest

load_dotenv()
LOG_FILE = os.getenv("SIGNAL_LOG", "signals_log.json")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def generate_signal(df: pd.DataFrame) -> pd.Series:
    """
    Composite signal: RSI, MACD diff, EMA diff, Bollinger Bands position.
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
        # Return a zero series as fallback
        return pd.Series(0, index=df.index)


def smooth_signal(signal: pd.Series, window: int = 5) -> pd.Series:
    """Rolling mean to smooth noise."""
    return signal.rolling(window, min_periods=1).mean()


def adaptive_threshold(
    df: pd.DataFrame,
    target_profit: float = 0.0,
    n_steps: int = 20
) -> float:
    """
    Dynamically choose a threshold that:
      - Actually produces trades in backtest
      - Maximizes average return
    Falls back to a small threshold if nothing trades.
    """
    # First generate+smooth the signal series
    sig = smooth_signal(generate_signal(df)).fillna(0)
    abs_max = sig.abs().max()

    # build threshold grid from 1% of max up to max
    if abs_max <= 0:
        return 0.0
    thresholds = np.linspace(abs_max * 0.01, abs_max, n_steps)

    best_t, best_ret = thresholds[0], -np.inf
    for t in thresholds:
        bt = run_backtest(sig, df["Close"], threshold=t)
        if bt.empty:
            continue
        avg_ret = bt["return"].mean()
        logging.info(f"[Adaptive] t={t:.4f}, trades={len(bt)}, avg_ret={avg_ret:.4%}")
        if avg_ret > best_ret:
            best_ret, best_t = avg_ret, t

    if best_ret == -np.inf:
        logging.warning(
            f"⚠️ adaptive_threshold: no trades at any threshold; "
            f"using fallback {thresholds.min():.4f}"
        )
        return float(thresholds.min())

    logging.info(f"✨ adaptive_threshold chosen={best_t:.4f} (avg_ret={best_ret:.4%})")
    return float(best_t)


def track_trade_result(response: dict, pair: str, action: str):
    """
    Append a JSON record to disk for auditing.
    """
    rec = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "pair":      pair,
        "action":    action,
        "response":  response
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
