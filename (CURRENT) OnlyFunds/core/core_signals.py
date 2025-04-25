import pandas as pd
import numpy as np
import logging
from core.backtester import run_backtest

const COMMON_QUOTES = ["USDT", "BTC", "ETH", "BNB"]  # used if/when inferring symbols


def generate_signal(df):
    """
    Generate trading signals based on given indicators in the DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing indicator columns.

    Returns:
    - pd.Series: A series of raw trading signals (e.g., [-1, 0, 1]).
    """
    # Guard against missing 'indicator' column
    if "indicator" not in df.columns:
        logging.error(
            f"[generate_signal] Missing 'indicator' column. Available columns: {df.columns.tolist()}"
        )
        # Return a zero signal series to prevent KeyError
        return pd.Series(0, index=df.index)

    # Compute raw signal: +1 when indicator > 0, -1 when < 0, 0 otherwise
    return (df["indicator"] > 0).astype(int) - (df["indicator"] < 0).astype(int)


def smooth_signal(signal, smoothing_window=5):
    """
    Smooth the trading signal using a moving average.

    Parameters:
    - signal (pd.Series): The raw trading signal.
    - smoothing_window (int): The window size for the moving average.

    Returns:
    - pd.Series: The smoothed trading signal.
    """
    return signal.rolling(window=smoothing_window).mean().fillna(0)


def adaptive_threshold(df, target_profit=0.01):
    """
    Find the optimal trading signal threshold for maximizing average return.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing price and indicator data.
    - target_profit (float): The desired target average return.

    Returns:
    - float: The optimal threshold value.
    """
    best_t, best_r = 0.5, -float("inf")

    # Smooth the series, guarding if generate_signal returns zeros
    sig = smooth_signal(generate_signal(df))

    # Safely get close prices; default to zeros if missing
    prices = df.get("Close") if "Close" in df.columns else pd.Series(0, index=df.index)

    for t in np.arange(0.1, 1.0, 0.05):
        summary_df, _ = run_backtest(sig, prices, threshold=t)
        if "avg_return" not in summary_df.columns:
            logging.warning(
                f"[adaptive_threshold] Missing 'avg_return' in backtest summary: {summary_df.columns.tolist()}"
            )
            continue

        avg = summary_df.at[0, "avg_return"]
        if avg > best_r:
            best_r, best_t = avg, t

    return best_t


def track_trade_result(result, pair, action):
    """
    Log the result of a trade and take any necessary post-trade actions.

    Parameters:
    - result (dict): The result of the trade execution.
    - pair (str): The trading pair (e.g., BTC/USDT).
    - action (str): The action taken (e.g., "BUY" or "SELL").
    """
    if not result.get("filled", False):
        logging.warning(f"Trade for {pair} ({action}) was not filled.")
        return

    logging.info(
        f"Trade for {pair} ({action}) filled: "
        f"Order ID: {result.get('order_id')}, Amount: {result.get('amount')}, "
        f"Price: {result.get('order_price')}"
    )
