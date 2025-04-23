import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def run_backtest(signals: pd.Series, prices: pd.Series, threshold: float = 0.2) -> tuple:
    """
    Backtests trading strategy for SPOT market.
    Entry on signal > threshold (buy).
    Exit when signal crosses back to neutral zone (close).
    """
    entry_price = 0
    returns = []
    trade_log = []
    position_open = False

    for i in range(1, len(signals)):
        signal = signals.iloc[i]
        price = prices.iloc[i]

        # Log signal and threshold comparison
        logger.info(f"Signal={signal}, Threshold={threshold}, Position Open={position_open}")

        if not position_open and signal > threshold:  # Open a long position
            entry_price = price
            position_open = True
        elif position_open and signal < threshold:  # Close the long position
            ret = (price - entry_price) / entry_price
            returns.append(ret)
            trade_log.append({
                "entry": entry_price,
                "exit": price,
                "side": "CLOSE",
                "return": round(ret, 5)
            })
            position_open = False

    if not returns:
        logger.warning("⚠️ No trades executed during backtesting. Check your signals or thresholds.")
        return pd.DataFrame([{
            "trades": 0,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0
        }]), pd.DataFrame()

    returns = np.array(returns)
    cum_returns = np.cumsum(returns)
    rolling_max = np.maximum.accumulate(cum_returns)
    drawdown = rolling_max - cum_returns
    max_dd = drawdown.max()

    win_rate = (returns > 0).mean()
    avg_return = returns.mean()
    std_return = returns.std() if returns.std() > 0 else 1e-8
    sharpe_ratio = avg_return / std_return

    # Summary metrics
    summary = pd.DataFrame([{
        "trades": len(returns),
        "win_rate": round(win_rate * 100, 2),
        "avg_return": round(avg_return * 100, 2),
        "sharpe": round(sharpe_ratio, 2),
        "max_drawdown": round(max_dd * 100, 2)
    }])
    return summary, pd.DataFrame(trade_log)
