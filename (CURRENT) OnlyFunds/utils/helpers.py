# utils/helpers.py

import os
import json
import time
import random
import string
from typing import List, Dict, Any
import pandas as pd
import numpy as np

# —————————————————————————————————————————————————————————————————————
# State-storage utilities
# —————————————————————————————————————————————————————————————————————

def save_json(path: str, data: Any) -> None:
    """Save any JSON-serializable object to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(path: str) -> Any:
    """Load a JSON file from disk, returning Python data."""
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)

# —————————————————————————————————————————————————————————————————————
# Symbol / rate-limit helpers
# —————————————————————————————————————————————————————————————————————

def validate_pair(pair: str, allowed: List[str]) -> bool:
    """Return True if `pair` is in our allowed list (case-insensitive)."""
    return pair.upper() in [p.upper() for p in allowed]

def check_rate_limit(last_call: float, cooldown: float) -> bool:
    """
    Given a timestamp of the last call and a cooldown in seconds,
    return True if we’re allowed to fire again now.
    """
    return (time.time() - last_call) >= cooldown

def format_timestamp(ts: float) -> str:
    """Convert a Unix timestamp to an ISO‐8601 string."""
    return pd.to_datetime(ts, unit='s').isoformat()

def generate_random_string(length: int = 8) -> str:
    """Produce a random alphanumeric string (for e.g. cache keys)."""
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))


# —————————————————————————————————————————————————————————————————————
# Performance metrics & tuning suggestions
# —————————————————————————————————————————————————————————————————————

def calculate_performance(
    trade_log: List[Dict[str, Any]],
    initial_capital: float = 1.0
) -> Dict[str, float]:
    """
    Given a list of trade records and starting capital, return key performance metrics:
      - total_return: (final_balance / initial_capital) - 1
      - win_rate: wins / total_trades
      - average_return: mean of individual trade returns
      - max_drawdown: maximum peak-to-trough drawdown
      - sharpe_ratio: (mean_return / std_return) * sqrt(N)
    Each trade record should contain at least 'entry_price', 'exit_price', 'amount', 'action', 'timestamp'.
    """
    if not trade_log:
        return dict.fromkeys([
            "total_return", "win_rate", "average_return",
            "max_drawdown", "sharpe_ratio"
        ], 0.0)

    df = pd.DataFrame(trade_log)
    # Per-trade return
    df["return_pct"] = np.where(
        df["action"].str.upper() == "BUY",
        (df["exit_price"] - df["entry_price"]) / df["entry_price"],
        (df["entry_price"] - df["exit_price"]) / df["entry_price"],
    )

    # Equity curve
    equity = initial_capital * (1 + df["return_pct"].cumsum())
    peak = equity.cummax()
    drawdown = (equity - peak) / peak

    total_return    = (equity.iloc[-1] / initial_capital) - 1
    win_rate        = (df["return_pct"] > 0).mean()
    average_return  = df["return_pct"].mean()
    max_drawdown    = drawdown.min()
    std             = df["return_pct"].std(ddof=0)
    sharpe_ratio    = (average_return / std) * np.sqrt(len(df)) if std and len(df) > 1 else 0.0

    return {
        "total_return": total_return,
        "win_rate": win_rate,
        "average_return": average_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio
    }


def suggest_tuning_parameters(metrics: Dict[str, float]) -> List[str]:
    """
    Given performance metrics, return a list of human-readable tuning suggestions.
    """
    suggestions = []
    if metrics.get("win_rate", 0) < 0.5:
        suggestions.append("Win rate below 50% → consider increasing entry threshold.")
    if metrics.get("max_drawdown", 0) < -0.05:
        suggestions.append("Max drawdown >5% → consider tightening stop-loss or reducing position size.")
    if metrics.get("sharpe_ratio", 0) > 1.5:
        suggestions.append("High Sharpe ratio → you may lower threshold to capture more trades.")
    if metrics.get("total_return", 0) < 0:
        suggestions.append("Overall negative return → review indicator parameters or switch to conservative mode.")
    if not suggestions:
        suggestions.append("Performance stable → maintain current strategy parameters.")

    return suggestions
