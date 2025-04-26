import pandas as pd
import numpy as np
import json
import os
import time
import string
import random
from datetime import datetime

def compute_trade_metrics(trade_log, initial_capital):
    df = pd.DataFrame(trade_log)
    required = {"entry_price", "exit_price"}
    if df.empty or not required.issubset(df.columns):
        return {
            "total_return": 0,
            "win_rate": 0,
            "average_return": 0,
            "max_drawdown": 0,
            "sharpe_ratio": 0
        }
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
        "total_return": total_return * 100,
        "win_rate": win_rate,
        "average_return": average_return * 100,
        "max_drawdown": max_drawdown * 100,
        "sharpe_ratio": sharpe_ratio
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
    fills = df[~df["fill_price"].isna()]
    total_pnl = 0
    if not fills.empty and "side" in fills.columns:
        buy_fills = fills[fills["side"] == "buy"]
        sell_fills = fills[fills["side"] == "sell"]
        if not buy_fills.empty and not sell_fills.empty:
            # Simple PnL: sum of (sell - buy) * size for matched pairs
            min_fills = min(len(buy_fills), len(sell_fills))
            total_pnl = ((sell_fills["fill_price"].iloc[:min_fills].values -
                         buy_fills["fill_price"].iloc[:min_fills].values) *
                         buy_fills["size"].iloc[:min_fills].values).sum()
        else:
            total_pnl = 0
        return {
            "total_pnl": total_pnl,
            "fills": len(fills),
            "buy_fills": len(buy_fills) if not buy_fills.empty else 0,
            "sell_fills": len(sell_fills) if not sell_fills.empty else 0,
            "avg_fill_price": fills["fill_price"].mean() if not fills.empty else 0,
        }
    else:
        return {
            "total_pnl": 0,
            "fills": 0,
            "buy_fills": 0,
            "sell_fills": 0,
            "avg_fill_price": 0,
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