import pandas as pd
import numpy as np
import json
import os
import time
import string
import random
from datetime import datetime

def save_json(data, filepath, **json_kwargs):
    """Serialize `data` to a JSON file at `filepath`."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, **json_kwargs)

def load_json(filepath):
    """Load and return JSON data from `filepath`."""
    with open(filepath, "r") as f:
        return json.load(f)

def validate_pair(pair: str):
    """
    Ensure a symbol pair like "BTC/USDT".
    Returns (base, quote) uppercased.
    """
    if not isinstance(pair, str) or "/" not in pair:
        raise ValueError(f"Invalid trading pair: {pair!r}")
    base, quote = pair.split("/", 1)
    return base.upper(), quote.upper()

def check_rate_limit(last_call_ts: float, min_interval: float = 1.0):
    """
    Enforce a minimum interval (in seconds) between calls.
    Returns the updated timestamp.
    """
    elapsed = time.time() - last_call_ts
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    return time.time()

def format_timestamp(ts: float, fmt: str = "%Y-%m-%d %H:%M:%S"):
    """Convert UNIX timestamp `ts` to formatted string."""
    return datetime.fromtimestamp(ts).strftime(fmt)

def generate_random_string(length: int = 8):
    """Return a random alphanumeric string of given length."""
    chars = string.ascii_letters + string.digits
    return "".join(random.choice(chars) for _ in range(length))

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
    df = pd.DataFrame(trade_log)
    df = df.dropna(subset=["entry_price", "exit_price"])
    if df.empty:
        return {
            "total_return": 0,
            "win_rate": 0,
            "average_return": 0,
            "max_drawdown": 0,
            "sharpe_ratio": 0
        }

    df["trade_return"] = (df["exit_price"] - df["entry_price"]) / df["entry_price"]
    final_capital = initial_capital * (1 + df["trade_return"]).prod()
    total_return = (final_capital - initial_capital) / initial_capital
    win_rate = (df["trade_return"] > 0).mean() * 100
    average_return = df["trade_return"].mean()
    cumulative_returns = (1 + df["trade_return"]).cumprod()
    drawdown = 1 - cumulative_returns / cumulative_returns.cummax()
    max_drawdown = drawdown.max()
    sharpe_ratio = df["trade_return"].mean() / df["trade_return"].std() if df["trade_return"].std() != 0 else 0

    return {
        "total_return": total_return * 100,  # percent
        "win_rate": win_rate,
        "average_return": average_return * 100,  # percent
        "max_drawdown": max_drawdown * 100,  # percent
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
