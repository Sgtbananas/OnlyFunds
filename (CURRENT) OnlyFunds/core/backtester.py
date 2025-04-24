import pandas as pd
import numpy as np
import logging

def run_backtest(
    signal: pd.Series,
    prices: pd.Series,
    threshold: float = 0.05,
    initial_capital: float = 1000.0,
    verbose: bool = False,  # Toggle for detailed logging
    log_every_n: int = 50   # Only log every N steps if verbose is True
) -> pd.DataFrame:
    """
    Backtest a signal with a given threshold and return trade results.

    Parameters:
    - signal (pd.Series): The trading signal, smoothed and clipped to [-1, 1].
    - prices (pd.Series): The corresponding prices to trade.
    - threshold (float): The signal strength threshold for entering trades.
    - initial_capital (float): Starting capital for the backtest.
    - verbose (bool): Enable detailed step-by-step logging.
    - log_every_n (int): Log every N steps if verbose is enabled.

    Returns:
    - pd.DataFrame: A DataFrame of trade results with columns:
        - entry_price: The price at which the position was entered.
        - exit_price: The price at which the position was exited.
        - return_pct: The percentage return for the trade.
    """
    trades = []
    position = None  # Tracks the current position
    entry_price = None

    for i in range(len(signal)):
        sig = signal.iloc[i]
        price = prices.iloc[i]

        # Optional verbose logging
        if verbose and (i % log_every_n == 0):
            logging.debug(f"Step {i}: Signal={sig:.4f}, Price={price:.2f}, Position={position}")

        # Handle LONG entry
        if sig > threshold and position is None:
            position = "long"
            entry_price = price
            logging.info(f"Entered LONG at {entry_price:.2f}")

        # Handle LONG exit
        elif sig < 0 and position == "long":
            exit_price = price
            return_pct = (exit_price - entry_price) / entry_price
            trades.append({
                "entry_price": entry_price,
                "exit_price": exit_price,
                "return_pct": return_pct
            })
            logging.info(f"Exited LONG at {exit_price:.2f} → Return: {return_pct:.2%}")
            position = None  # Reset position

    # Convert trade results to a DataFrame
    trade_results = pd.DataFrame(trades)

    if trade_results.empty:
        logging.warning("No trades executed during backtest.")
    else:
        avg_return = trade_results["return_pct"].mean()
        logging.info(f"Backtest complete: {len(trade_results)} trades, Avg Return: {avg_return:.2%}")

    return trade_results
