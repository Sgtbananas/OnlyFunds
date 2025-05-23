import pandas as pd
import numpy as np
import json
import os
import time
import string
import random
from datetime import datetime
from sklearn.linear_model import LinearRegression
import logging

logger = logging.getLogger(__name__)

# Blacklisted tokens to exclude from trading
BLACKLISTED_TOKENS = {
    "TAPUSDT", "ARTYUSDT", "BROCCOLIUSDT", "BOOPUSDT",
    "ZEROUSDT", "PSPUSDT", "CWIFUSDT", "AZEROUSDT", "BITCOINUSDT"
}

def get_volatile_pairs(limit=10, interval="1h", market="USDT", min_volume=50000, min_change=1.0):
    """
    Return the top volatile pairs meeting volume and change requirements,
    excluding blacklisted tokens.
    """
    import requests
    try:
        url = "https://api.coinex.com/v1/market/ticker/all"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()["data"]["ticker"]

        candidates = []
        for pair, stats in data.items():
            if not pair.endswith(market):
                continue
            if pair in BLACKLISTED_TOKENS:
                continue
            try:
                volume = float(stats.get("vol", 0))
                change_pct = abs(float(stats.get("percent_change_24h", 0)))
                if volume >= min_volume and change_pct >= min_change:
                    score = change_pct * volume
                    candidates.append((pair, score))
            except Exception:
                continue

        sorted_pairs = sorted(candidates, key=lambda x: x[1], reverse=True)
        return [p[0] for p in sorted_pairs[:limit]]
    except Exception as e:
        logger.error(f"get_volatile_pairs failed: {e}")
        return ["BTCUSDT", "ETHUSDT", "LTCUSDT"]
def validate_trade(amount, price, capital):
    estimated_cost = amount * price
    return capital >= estimated_cost * 1.01  # Allow buffer for fees

def compute_trade_metrics(trade_log, initial_capital):
    if not trade_log:
        return {
            "total_return": 0,
            "win_rate": 0,
            "average_return": 0,
            "max_drawdown": 0,
            "sharpe_ratio": 0,
            "current_capital": initial_capital
        }

    df = pd.DataFrame(trade_log)
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
    total_return = (capital / initial_capital - 1) * 100
    average_return = returns.mean() * 100

    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = 1 - cumulative / peak
    max_drawdown = drawdown.max() * 100 if not drawdown.empty else 0

    sharpe_ratio = (returns.mean() / returns.std()) if returns.std() != 0 else 0

    return {
        "total_return": total_return,
        "win_rate": min(max(win_rate, 0), 100),
        "average_return": average_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "current_capital": capital
    }
def compute_grid_metrics(grid_orders):
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
    tmpfile = filepath + ".tmp"
    with open(tmpfile, "w") as f:
        json.dump(data, f, **json_kwargs)
    os.replace(tmpfile, filepath)

def load_json(filepath, default=None):
    try:
        if not os.path.exists(filepath):
            return default
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON from {filepath}: {e}")
        return default
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

def dynamic_threshold(df):
    """
    Estimate a dynamic buy threshold based on recent volatility.
    Higher volatility → require stronger signals.
    Lower volatility → accept weaker signals.
    """
    try:
        returns = df["Close"].pct_change()
        vol = returns.std()

        if vol > 0.02:
            return 0.7
        elif vol > 0.01:
            return 0.6
        else:
            return 0.5
    except Exception:
        return 0.5  # fallback

def estimate_dynamic_atr_multipliers(df):
    """
    Simple dynamic tuning of ATR-based stop loss, take profit, and trailing multipliers.
    """
    try:
        if "ATR" not in df.columns:
            raise ValueError("ATR missing from dataframe.")

        atr = df["ATR"].dropna()
        volatility = df["Close"].pct_change().rolling(20).std().dropna()

        stop_mult = 1.5
        tp_mult = 2.5
        trail_mult = 1.2

        if not atr.empty and not volatility.empty:
            avg_atr = atr.mean()
            avg_vol = volatility.mean()

            stop_mult = max(0.8, min(2.0, 1.5 + avg_vol * 10))
            tp_mult = max(1.5, min(4.0, 2.5 + avg_vol * 20))
            trail_mult = max(0.8, min(2.5, 1.0 + avg_vol * 10))

        return stop_mult, tp_mult, trail_mult

    except Exception as e:
        print(f"[WARN] estimate_dynamic_atr_multipliers failed: {e}")
        return 1.5, 2.5, 1.2

def estimate_optimal_threshold(df, signal, prices, n_steps=20, risk_pct=0.01, fee_pct=0.001):
    """
    Automatically finds the best entry threshold for signal strength.
    """
    try:
        thresholds = np.linspace(0.3, 0.8, n_steps)
        best_thr = 0.5
        best_profit = -np.inf

        for thr in thresholds:
            buys = (signal > thr)
            sells = (signal < -thr)
            profits = []

            for i in range(1, len(prices)):
                if buys.iloc[i - 1]:
                    entry = prices.iloc[i - 1]
                    exit = prices.iloc[i]
                    ret = (exit - entry) / entry - fee_pct
                    profits.append(ret * risk_pct)
                elif sells.iloc[i - 1]:
                    entry = prices.iloc[i - 1]
                    exit = prices.iloc[i]
                    ret = (entry - exit) / entry - fee_pct
                    profits.append(ret * risk_pct)

            avg_profit = np.mean(profits) if profits else -999
            if avg_profit > best_profit:
                best_profit = avg_profit
                best_thr = thr

        return best_thr
    except Exception as e:
        print(f"[WARN] estimate_optimal_threshold fallback: {e}")
        return 0.5
