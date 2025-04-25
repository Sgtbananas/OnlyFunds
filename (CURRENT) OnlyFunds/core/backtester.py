import pandas as pd
import numpy as np
import logging

def run_backtest(
    signal: pd.Series,
    prices: pd.Series,
    threshold: float = 0.05,
    initial_capital: float = 10.0,
    risk_pct: float = 0.01,  # Risk percentage per trade
    verbose: bool = False,  # Toggle for detailed logging
    log_every_n: int = 50   # Only log every N steps if verbose is True
) -> pd.DataFrame:
    """
    Backtest a signal with a given threshold and return a combined DataFrame.

    Parameters:
    - signal (pd.Series): The trading signal, smoothed and clipped to [0, 1].
    - prices (pd.Series): The corresponding prices to trade.
    - threshold (float): The signal strength threshold for entering trades.
    - initial_capital (float): Starting capital for the backtest.
    - risk_pct (float): The percentage of capital to risk per trade.
    - verbose (bool): Enable detailed step-by-step logging.
    - log_every_n (int): Log every N steps if verbose is enabled.

    Returns:
    - pd.DataFrame: A single DataFrame containing both summary metrics and trade details.
      Summary metrics are included as the first row with the column "type" set to "summary".
      Trade details follow with the column "type" set to "trade".
    """
    trades = []
    position = None  # Tracks the current position (only long)
    entry_price = None
    capital = initial_capital

    for i in range(len(signal)):
        sig = signal.iloc[i]
        price = prices.iloc[i]

        # Optional verbose logging
        if verbose and (i % log_every_n == 0):
            logging.debug(f"Step {i}: Signal={sig:.4f}, Price={price:.2f}, Position={position}, Capital={capital:.2f}")

        # Handle LONG entry
        if sig > threshold and position is None:  # Enter a long position
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
        elif sig < threshold and position is not None:  # Exit the long position
            exit_price = price
            position_size = position["size"]
            return_pct = (exit_price - entry_price) / entry_price
            profit = position_size * (exit_price - entry_price)
            capital += position_size * exit_price  # Add position value back to capital
            trades.append({
                "type": "trade",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "return": return_pct,
                "profit": profit,
                "capital": capital,
            })
            logging.info(f"Exited LONG at {exit_price:.2f} → Return: {return_pct:.2%}, Capital: {capital:.2f}")
            position = None  # Reset position

    # Convert trade results to a DataFrame
    trades_df = pd.DataFrame(trades)

    # Compute summary metrics
    summary = {
        "type": "summary",
        "trades": len(trades_df),
        "avg_return": trades_df["return"].mean() if not trades_df.empty else 0,
        "win_rate": (trades_df["return"] > 0).mean() * 100 if not trades_df.empty else 0,
        "capital": capital,
    }
    summary_df = pd.DataFrame([summary])

    # Combine summary and trades into a single DataFrame
    combined_df = pd.concat([summary_df, trades_df], ignore_index=True)

    if trades_df.empty:
        logging.warning("No trades executed during backtest.")
    else:
        avg_return = summary["avg_return"]
        logging.info(f"Backtest complete: {summary['trades']} trades, Avg Return: {avg_return:.2%}")

    return combined_df
