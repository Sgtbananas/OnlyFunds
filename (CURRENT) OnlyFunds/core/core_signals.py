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

def adaptive_threshold(
    df: pd.DataFrame,
    signal_col: str = "signal",      # You might generate_signal(df) into df["signal"]
    target_profit: float = 0.0,      # We just look for any positive avg return
    thresholds: np.ndarray = None   # Allow injecting a custom grid
) -> float:
    """
    Dynamically choose a threshold that:
      - Actually produces trades in backtest
      - Maximizes average return
    """
    try:
        if thresholds is None:
            # Scan from very small up to the max absolute signal
            signals = df[signal_col].fillna(0).abs()
            max_sig = signals.max()
            # Scan 20 steps between 1% of max to 100% of max
            thresholds = np.linspace(max_sig * 0.01, max_sig, 20)

        best_t = thresholds[0]
        best_ret = -np.inf

        for t in thresholds:
            bt = run_backtest(df[signal_col], df["Close"], threshold=t)
            if bt.empty:
                continue
            avg_ret = bt["return"].mean()
            logging.info(f"Threshold={t:.4f} → trades={len(bt)}, avg_return={avg_ret:.4%}")
            if avg_ret > best_ret:
                best_ret = avg_ret
                best_t = t

        if best_ret == -np.inf:
            logging.warning(
                "⚠️ adaptive_threshold: no trades for any threshold, "
                f"falling back to 0.01 (signal max={thresholds.max():.4f})"
            )
            return thresholds.min()  # Something tiny so you at least trade
        logging.info(f"✨ Chosen threshold = {best_t:.4f} (avg_return={best_ret:.4%})")
        return float(best_t)

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
