import pandas as pd
import numpy as np
import json
import os
import time
import string
import random
from datetime import datetime

def compute_trade_metrics(trade_log, initial_capital):
    # Defensive: If empty, return zeroed metrics
    if not trade_log:
        return {
            "total_return": 0,
            "win_rate": 0,
            "average_return": 0,
            "max_drawdown": 0,
            "sharpe_ratio": 0,
            "current_capital": initial_capital
        }
    # Try to use return_pct if present, else fallback to entry/exit
    df = pd.DataFrame(trade_log)
    # Use return_pct if present on all trades, else fallback to calculated trade_return
    if "return_pct" in df.columns and not df["return_pct"].isnull().all():
        returns = df["return_pct"].dropna()
    elif {"entry_price", "exit_price"}.issubset(df.columns):
        returns = ((df["exit_price"] - df["entry_price"]) / df["entry_price"]).dropna()
    else:
        returns = pd.Series(dtype=float)

    if returns.empty:
        return {
            "total_return": 0,
            "win_rate": 0,
            "average_return": 0,
            "max_drawdown": 0,
            "sharpe_ratio": 0,
            "current_capital": initial_capital
        }

    capital = initial_capital
    wins = 0
    for r in returns:
        capital *= (1 + r)
        if r > 0:
            wins += 1

    trades = len(returns)
    win_rate = (wins / trades * 100) if trades else 0
    win_rate = min(max(win_rate, 0), 100)  # Clamp 0-100%
    total_return = (capital / initial_capital - 1) * 100
    total_return = max(total_return, -100)  # Clamp to -100% max loss
    average_return = returns.mean() * 100

    # Max drawdown: in % (peak-to-valley)
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = 1 - cumulative / peak
    max_drawdown = drawdown.max() * 100 if not drawdown.empty else 0

    sharpe_ratio = (returns.mean() / returns.std()) if returns.std() != 0 else 0

    return {
        "total_return": total_return,
        "win_rate": win_rate,
        "average_return": average_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "current_capital": capital
    }

def compute_grid_metrics(grid_orders):
    """Compute metrics for grid trading orders (PnL, fill rates, etc)."""
    df = pd.DataFrame(grid_orders)
    if df.empty or "fill_price" not in df.columns or "side" not in df.columns:
        return {
            "total_pnl": 0,
            "fills": 0,
            "buy_fills": 0,
            "sell_fills": 0,
            "avg_fill_price": 0,
        }
    fills = df[df["filled"]]
    total_pnl = 0
    buy_fills = fills[fills["side"] == "buy"]
    sell_fills = fills[fills["side"] == "sell"]
    min_fills = min(len(buy_fills), len(sell_fills))
    if min_fills > 0:
        total_pnl = ((sell_fills["fill_price"].iloc[:min_fills].values -
                     buy_fills["fill_price"].iloc[:min_fills].values) *
                     buy_fills["size"].iloc[:min_fills].values).sum()
    return {
        "total_pnl": total_pnl,
        "fills": len(fills),
        "buy_fills": len(buy_fills),
        "sell_fills": len(sell_fills),
        "avg_fill_price": fills["fill_price"].mean() if not fills.empty else 0,
    }

def suggest_tuning(trade_log):
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

def save_json(data, filepath, **json_kwargs):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, **json_kwargs)

def load_json(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

def validate_pair(pair: str):
    if not isinstance(pair, str):
        raise ValueError(f"Invalid trading pair: {pair!r}")
    p = pair.strip().upper()
    if "/" in p:
        base, quote = p.split("/", 1)
    else:
        known_quotes = ["USDT", "BTC", "ETH", "BNB"]
        for q in known_quotes:
            if p.endswith(q) and len(p) > len(q):
                base = p[:-len(q)]
                quote = q
                break
        else:
            raise ValueError(f"Invalid trading pair: {pair!r}")
    if not base or not quote:
        raise ValueError(f"Invalid trading pair: {pair!r}")
    return base, quote

def check_rate_limit(last_call_ts: float, min_interval: float = 1.0):
    elapsed = time.time() - last_call_ts
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    return time.time()

def format_timestamp(ts: float, fmt: str = "%Y-%m-%d %H:%M:%S"):
    return datetime.fromtimestamp(ts).strftime(fmt)

def generate_random_string(length: int = 8):
    chars = string.ascii_letters + string.digits
    return "".join(random.choice(chars) for _ in range(length))

def log_grid_trade(trade_data, log_file="data/logs/grid_trade_logs.json"):
    """Append a grid trade to a dedicated grid trade log."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    trade_data["timestamp"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    try:
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                logs = json.load(f)
        else:
            logs = []
        logs.append(trade_data)
        with open(log_file, "w") as f:
            json.dump(logs, f, indent=2)
    except Exception as e:
        print(f"Error logging grid trade: {e}")