import pandas as pd
import numpy as np
import json
import os
import time
import string
import random
from datetime import datetime, date
import threading


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
    win_rate = min(max(win_rate, 0), 100)
    total_return = (capital / initial_capital - 1) * 100
    total_return = max(total_return, -100)
    average_return = returns.mean() * 100

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
    try:
        os.replace(tmpfile, filepath)
    except Exception:
        # fallback for Windows
        os.remove(filepath)
        os.rename(tmpfile, filepath)

import json
import os

def load_json(filepath, default=None):
    try:
        if not os.path.exists(filepath):
            return default
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load JSON from {filepath}: {e}")
        return default

def get_pair_params(pair):
    """
    Load dynamic params (interval, lookback, threshold) for a given trading pair.
    Fallback to default if not found.
    """
    try:
        params = load_json("state/auto_params.json")
    except Exception:
        params = {}

    if params and pair in params:
        return params[pair]
    
    return dict(
        interval="5m",
        lookback=1000,
        threshold=0.5
    )


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


def get_auto_pair_params(auto_params, pair, today=None, fallback=None):
    """
    Find the most recent available params for a given pair and date in auto_params.json.
    If today's params not found, fallback to the latest available or default.
    """
    if today is None:
        today = date.today()
    
    if pair not in auto_params:
        return fallback
    
    pair_params = auto_params[pair]
    today_str = str(today)
    if today_str in pair_params:
        return pair_params[today_str]
    
    # Fallback: get latest available before today
    dates = sorted([d for d in pair_params if d <= today_str])
    if dates:
        return pair_params[dates[-1]]
    
    return fallback


def get_pair_params(pair):
    """
    Load dynamic params (interval, lookback, threshold) for a trading pair.
    Fallback: auto-tune interval based on volatility.
    """
    try:
        params = load_json("state/auto_params.json")
    except Exception:
        params = {}

    if params and pair in params:
        return params[pair]

    # --- Volatility-based fallback ---
    try:
        df = fetch_klines(pair, interval="5m", limit=300)
        if df.empty:
            raise ValueError("Empty dataframe")

        df["returns"] = df["Close"].pct_change()
        volatility = df["returns"].std()

        # More volatile pairs = slower intervals
        if volatility > 0.02:
            interval = "15m"
            lookback = 800
        elif volatility > 0.01:
            interval = "5m"
            lookback = 1000
        else:
            interval = "1m"
            lookback = 1200

        return dict(
            interval=interval,
            lookback=lookback,
            threshold=0.5  # fallback default threshold
        )

    except Exception as e:
        print(f"[WARN] Auto fallback get_pair_params failed for {pair}: {e}")
        return dict(
            interval="5m",
            lookback=1000,
            threshold=0.5
        )


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
        return 0.5  # fallback safe default


import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor


def estimate_dynamic_atr_multipliers(df):
    """
    AI/ML-driven estimator for optimal ATR Stop Loss, Take Profit, and Trailing multipliers.
    Based on recent price action and volatility.
    """
    if df is None or df.empty or "ATR" not in df or "Close" not in df:
        # Fallback to basic defaults
        return 1.0, 2.0, 1.0

    try:
        # Feature engineering
        features = pd.DataFrame({
            "atr": df["ATR"],
            "atr_pct_close": df["ATR"] / df["Close"],
            "rsi": df.get("rsi", pd.Series(50, index=df.index)),  # fallback RSI=50 if missing
            "volatility": df["Close"].pct_change().rolling(14).std().fillna(0)
        }).fillna(0)

        # Target variables: pseudo "ideal" stop/TP settings
        target_stop = np.clip(features["atr_pct_close"] * np.random.uniform(0.8, 1.2), 0.5, 3.0)
        target_tp = np.clip(features["atr_pct_close"] * np.random.uniform(1.5, 3.5), 1.5, 5.0)
        target_trail = np.clip(features["atr_pct_close"] * np.random.uniform(0.7, 1.5), 0.5, 3.0)

        # --- Fit Random Forest regressors
        X = features.values
        stop_model = RandomForestRegressor(n_estimators=30, max_depth=5, random_state=42)
        tp_model = RandomForestRegressor(n_estimators=30, max_depth=5, random_state=42)
        trail_model = RandomForestRegressor(n_estimators=30, max_depth=5, random_state=42)

        stop_model.fit(X, target_stop)
        tp_model.fit(X, target_tp)
        trail_model.fit(X, target_trail)

        # Predict using the last row (most recent candle)
        last_X = features.iloc[-1].values.reshape(1, -1)
        stop_mult = float(stop_model.predict(last_X)[0])
        tp_mult = float(tp_model.predict(last_X)[0])
        trail_mult = float(trail_model.predict(last_X)[0])

        # Clip outputs to sensible ranges
        stop_mult = np.clip(stop_mult, 0.5, 3.0)
        tp_mult = np.clip(tp_mult, 1.5, 5.0)
        trail_mult = np.clip(trail_mult, 0.5, 3.0)

        return stop_mult, tp_mult, trail_mult

    except Exception as e:
        print(f"[WARN] ATR estimation failed: {e}")
        return 1.0, 2.0, 1.0  # Safe default


import numpy as np


def estimate_optimal_threshold(df, signal, prices, n_steps=20, risk_pct=0.01, fee_pct=0.001):
    """
    AI-driven: Automatically finds the best entry threshold for signal strength.
    Based on max average profit per trade.
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


import numpy as np
from sklearn.linear_model import LinearRegression


def estimate_dynamic_atr_multipliers(df, window=50):
    """
    Smarter AI/ML dynamic tuning of ATR-based stop loss, take profit, and trailing multipliers.
    - Uses historical volatility and reward/risk optimization.
    """
    try:
        if "ATR" not in df.columns:
            raise ValueError("ATR missing from dataframe.")

        df = df.dropna().copy()

        atr_mean = df["ATR"].rolling(window).mean()
        volatility = df["Close"].pct_change().rolling(window).std()

        # Normalize ATR to Close price
        atr_norm = atr_mean / df["Close"]

        # Prepare simple features
        X = np.column_stack([
            atr_norm.fillna(0).values,
            volatility.fillna(0).values
        ])

        # Target: try to predict large candle movements (future volatility proxy)
        y = df["Close"].pct_change(periods=5).abs().shift(-5).fillna(0).values
        model = LinearRegression()
        model.fit(X, y)

        pred = model.predict(X)

        avg_pred = np.mean(pred)

        # Now map prediction to ATR multipliers
        stop_mult = max(0.8, min(2.0, 1.5 - avg_pred * 5))
        tp_mult = max(1.5, min(4.0, 2.5 + avg_pred * 8))
        trail_mult = max(0.8, min(2.5, 1.0 + avg_pred * 4))

        return stop_mult, tp_mult, trail_mult

    except Exception as e:
        print(f"[WARN] estimate_dynamic_atr_multipliers fallback: {e}")
        # Fallback to sane defaults
        return 1.0, 2.0, 1.0
