import pandas as pd
import numpy as np
import logging

def run_backtest(
    signal: pd.Series,
    prices: pd.Series,
    threshold: float = 0.05,
    initial_capital: float = 10.0,
    risk_pct: float = 0.01,
    stop_loss_pct: float = 0.005,
    take_profit_pct: float = 0.01,
    fee_pct: float = 0.0004,
    verbose: bool = False,
    log_every_n: int = 50
) -> pd.DataFrame:
    """
    Backtest with support for stop-loss, take-profit, and fee modeling.
    """
    trades = []
    position = None
    entry_price = None
    capital = initial_capital

    for i in range(len(signal)):
        sig = signal.iloc[i]
        price = prices.iloc[i]

        # Apply fee to entry/exit
        adjusted_entry_price = price * (1 + fee_pct) if position is None else entry_price
        adjusted_exit_price = price * (1 - fee_pct) if position is not None else price

        if verbose and (i % log_every_n == 0):
            logging.debug(f"Step {i}: Signal={sig:.4f}, Price={price:.2f}, Position={position}, Capital={capital:.2f}")

        # --- LONG entry ---
        if sig > threshold and position is None:
            position_size = (capital * risk_pct) / adjusted_entry_price
            position = {
                "size": position_size,
                "entry_price": adjusted_entry_price,
                "bar": i
            }
            entry_price = adjusted_entry_price
            position_cost = position_size * adjusted_entry_price
            capital -= position_cost
            logging.info(f"Entered LONG at {entry_price:.2f}, Size: {position_size:.4f}, Capital: {capital:.2f}")

        # --- LONG exit conditions ---
        elif position is not None:
            unrealized = (adjusted_exit_price / entry_price) - 1
            stop = unrealized <= -stop_loss_pct
            take = unrealized >= take_profit_pct
            flip = sig < threshold

            # Exit on SL, TP or signal flip
            if stop or take or flip:
                exit_price = adjusted_exit_price
                position_size = position["size"]
                return_pct = (exit_price - entry_price) / entry_price
                profit = position_size * (exit_price - entry_price)
                capital += position_size * exit_price
                trades.append({
                    "type": "trade",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "reason": (
                        "stop_loss" if stop
                        else "take_profit" if take
                        else "signal_flip"
                    ),
                    "return": return_pct,
                    "profit": profit,
                    "capital": capital,
                })
                logging.info(f"Exited LONG at {exit_price:.2f} → Return: {return_pct:.2%}, Reason: {trades[-1]['reason']}, Capital: {capital:.2f}")
                position = None

    trades_df = pd.DataFrame(trades)
    summary = {
        "type": "summary",
        "trades": len(trades_df),
        "avg_return": trades_df["return"].mean() if not trades_df.empty else 0,
        "win_rate": (trades_df["return"] > 0).mean() * 100 if not trades_df.empty else 0,
        "capital": capital,
    }
    summary_df = pd.DataFrame([summary])
    combined_df = pd.concat([summary_df, trades_df], ignore_index=True)

    if trades_df.empty:
        logging.warning("No trades executed during backtest.")
    else:
        avg_return = summary["avg_return"]
        logging.info(f"Backtest complete: {summary['trades']} trades, Avg Return: {avg_return:.2%}")

    return combined_df