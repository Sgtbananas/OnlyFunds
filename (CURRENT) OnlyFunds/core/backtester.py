import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def run_backtest(
    signals: pd.Series,
    prices: pd.Series,
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Simulates trades:
      - Goes long when signal > threshold
      - Exits long when signal < 0
      - (No shorts in spot market)
    Returns a DataFrame of individual trade returns.
    """
    position_open = False
    entry_price   = 0.0
    returns       = []

    for step, (sig, price) in enumerate(zip(signals, prices), start=1):
        # Only debug‐level log so it won't spam by default
        logger.debug(f"Step {step}: Signal={sig:.4f}, Price={price:.2f}, Position Open={position_open}")

        if not position_open and sig > threshold:
            position_open = True
            entry_price   = price
            logger.debug(f"  → Entered LONG at {entry_price:.2f}")

        elif position_open and sig < 0:
            ret = (price - entry_price) / entry_price
            returns.append(ret)
            logger.debug(f"  → Closed LONG at {price:.2f}, Return={ret:.4%}")
            position_open = False

    # If nothing traded
    if not returns:
        logger.warning(
            "⚠️ No trades executed during backtesting. Check your signals or thresholds. "
            f"Threshold={threshold}, Signal range=({signals.min():.4f}, {signals.max():.4f})"
        )

    # Log a one‐line summary at INFO level
    else:
        avg_ret = np.mean(returns)
        win_rate = np.mean([1 for r in returns if r > 0])
        logger.info(
            f"Backtest summary: {len(returns)} trades, "
            f"avg return {avg_ret:.2%}, win_rate {win_rate:.2%}"
        )

    # Return a simple DataFrame
    return pd.DataFrame({"return": returns})
