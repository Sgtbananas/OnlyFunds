import pandas as pd
import numpy as np
import logging
from core.backtester import run_backtest

COMMON_QUOTES = ["USDT", "BTC", "ETH", "BNB"]

def generate_signal(df):
    if "indicator" not in df.columns:
        logging.error(
            f"[generate_signal] Missing 'indicator' column. Available columns: {df.columns.tolist()}"
        )
        return pd.Series(0, index=df.index)
    return (df["indicator"] > 0).astype(int) - (df["indicator"] < 0).astype(int)

def smooth_signal(signal, smoothing_window=5):
    return signal.rolling(window=smoothing_window).mean().fillna(0)

def adaptive_threshold(df, target_profit=0.01):
    best_t, best_r = 0.5, -float("inf")
    sig = smooth_signal(generate_signal(df))
    prices = df.get("Close") if "Close" in df.columns else pd.Series(0, index=df.index)
    for t in np.arange(0.1, 1.0, 0.05):
        combined_df = run_backtest(sig, prices, threshold=t)
        if "type" in combined_df.columns and combined_df.iloc[0].get("type") == "summary":
            summary_record = combined_df.iloc[0]
        else:
            summary_record = combined_df.iloc[0]
        avg = summary_record.get("avg_return")
        if avg is None:
            continue
        if avg > best_r:
            best_r, best_t = avg, t
    return best_t

def track_trade_result(result, pair, action):
    if not result.get("filled", False):
        logging.warning(f"Trade for {pair} ({action}) was not filled.")
        return
    logging.info(
        f"Trade for {pair} ({action}) filled: "
        f"Order ID: {result.get('order_id')}, Amount: {result.get('amount')}, "
        f"Price: {result.get('order_price')}"
    )
