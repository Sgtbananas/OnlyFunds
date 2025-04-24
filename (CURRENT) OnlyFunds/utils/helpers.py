import pandas as pd
import numpy as np

def compute_trade_metrics(trade_log, initial_capital):
    """
    Compute performance metrics for a trade log.

    Parameters:
    - trade_log (list of dict): The trade log containing records of trades.
    - initial_capital (float): The initial capital at the start of trading.

    Returns:
    - dict: A dictionary of performance metrics including:
        - total_return: Total return as a percentage.
        - win_rate: Percentage of winning trades.
        - average_return: Average return per trade.
        - max_drawdown: Maximum capital drawdown as a percentage.
        - sharpe_ratio: Sharpe ratio of the trade returns.
    """
    # Convert trade log to DataFrame
    df = pd.DataFrame(trade_log)

    # Only rows that have both entry_price and exit_price
    df = df.dropna(subset=["entry_price", "exit_price"])
    if df.empty:
        return {
            "total_return": 0,
            "win_rate": 0,
            "average_return": 0,
            "max_drawdown": 0,
            "sharpe_ratio": 0
        }

    # Calculate returns for each trade
    df["trade_return"] = (df["exit_price"] - df["entry_price"]) / df["entry_price"]

    # Total return
    final_capital = initial_capital * (1 + df["trade_return"]).prod()
    total_return = (final_capital - initial_capital) / initial_capital

    # Win rate
    win_rate = (df["trade_return"] > 0).mean() * 100

    # Average return
    average_return = df["trade_return"].mean()

    # Max drawdown
    cumulative_returns = (1 + df["trade_return"]).cumprod()
    drawdown = 1 - cumulative_returns / cumulative_returns.cummax()
    max_drawdown = drawdown.max()

    # Sharpe ratio
    sharpe_ratio = df["trade_return"].mean() / df["trade_return"].std() if df["trade_return"].std() != 0 else 0

    return {
        "total_return": total_return * 100,  # Convert to percentage
        "win_rate": win_rate,
        "average_return": average_return * 100,  # Convert to percentage
        "max_drawdown": max_drawdown * 100,  # Convert to percentage
        "sharpe_ratio": sharpe_ratio
    }

def suggest_tuning(trade_log):
    """
    Suggest tuning recommendations based on the trade log.

    Parameters:
    - trade_log (list of dict): The trade log containing records of trades.

    Returns:
    - dict: A dictionary of tuning suggestions.
    """
    if not trade_log:
        return {"suggestions": ["Insufficient data to provide tuning recommendations."]}

    df = pd.DataFrame(trade_log)

    # Suggestions based on performance
    suggestions = []
    if "return_pct" in df.columns:
        avg_return = df["return_pct"].mean()
        if avg_return < 0:
            suggestions.append("Consider reducing risk exposure or adjusting strategy.")
        elif avg_return > 0.05:
            suggestions.append("Strategy performing well. Consider scaling up.")
    else:
        suggestions.append("Unable to calculate return percentage for tuning.")

    if len(df) > 100:
        suggestions.append("Sufficient data for comprehensive analysis.")
    else:
        suggestions.append("Gather more data for better tuning insights.")

    return {"suggestions": suggestions}
