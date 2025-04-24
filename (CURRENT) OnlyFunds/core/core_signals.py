import logging
import numpy as np
import pandas as pd
from .core_data import fetch_klines
from .backtester import run_backtest
from .core_signals import smooth_signal, generate_signal

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
