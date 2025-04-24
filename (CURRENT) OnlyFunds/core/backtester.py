import pandas as pd
import numpy as np
import logging

def run_backtest(
    signal: pd.Series,
    prices: pd.Series,
    threshold: float = 0.05,
    initial_capital: float = 1000.0,
    risk_pct: float = 0.01,  # Risk percentage per trade
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
    - risk_pct (float): The percentage of capital to risk per trade.
    - verbose (bool): Enable detailed step-by-step logging.
    - log_every_n (int): Log every N steps if verbose is enabled.

    Returns:
    - pd.DataFrame: A DataFrame of trade results with columns:
        - entry_price: The price at which the position was entered.
        - exit_price: The price at which the position was exited.
        - return: The percentage return for the trade.
        - profit: The profit for the trade.
        - capital: The capital after each trade.
    """
    trades = []
    position = None  # Tracks the current position
    entry_price = None
    capital = initial_capital

    for i in range(len(signal)):
        sig = signal.iloc[i]
        price = prices.iloc[i]

        # Optional verbose logging
        if verbose and (i % log_every_n == 0):
            logging.debug(f"Step {i}: Signal={sig:.4f}, Price={price:.2f}, Position={position}, Capital={capital:.2f}")

        # Handle LONG entry
        if sig > threshold and position is None:
            # Compute position size based on available capital
            position_size = (capital * risk_pct) / price
            position = {
                "size": position_size,
                "entry_price": price,
            }
            entry_price = price
            position_cost = position_size * price
            capital -= position_cost  # Deduct position cost from capital
            logging.info(f"Entered LONG at {entry_price:.2f}, Size: {position_size:.4f}, Capital: {capital:.2f}")

        # Handle LONG exit
        elif sig < 0 and position is not None:
            exit_price = price
            position_size = position["size"]
            return_pct = (exit_price - entry_price) / entry_price
            profit = position_size * (exit_price - entry_price)
            capital += position_size * exit_price  # Add position value back to capital
            trades.append({
                "entry_price": entry_price,
                "exit_price": exit_price,
                "return": return_pct,  # Renamed from 'return_pct' for compatibility
                "profit": profit,
                "capital": capital,
            })
            logging.info(f"Exited LONG at {exit_price:.2f} → Return: {return_pct:.2%}, Capital: {capital:.2f}")
            position = None  # Reset position

    # Convert trade results to a DataFrame
    trade_results = pd.DataFrame(trades)

    if trade_results.empty:
        logging.warning("No trades executed during backtest.")
    else:
        avg_return = trade_results["return"].mean()  # Updated to use 'return'
        logging.info(f"Backtest complete: {len(trade_results)} trades, Avg Return: {avg_return:.2%}")

    return trade_results
