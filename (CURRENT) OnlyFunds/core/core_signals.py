import pandas as pd
import numpy as np
import logging
from core.backtester import run_backtest
from core.ml_filter import load_model, ml_confidence, extract_feature_array

COMMON_QUOTES = ["USDT", "BTC", "ETH", "BNB"]

def generate_signal(df):
    """
    Generate ensemble signals from multiple indicators:
    - RSI breakout
    - MACD cross
    - EMA momentum
    - Bollinger band squeeze & breakout
    Returns: pd.Series of signal strength (clipped -1..1)
    """
    sig = pd.Series(0.0, index=df.index)
    # RSI breakout: Strong buy when RSI crosses up from below 30, sell on 70 cross down
    if "rsi" in df.columns:
        sig += ((df["rsi"] > 30) & (df["rsi"].shift(1) <= 30)) * 0.5
        sig -= ((df["rsi"] < 70) & (df["rsi"].shift(1) >= 70)) * 0.5
    # MACD cross
    if "macd" in df.columns and "macd_signal" in df.columns:
        sig += (df["macd"] > df["macd_signal"]) * 0.3
        sig -= (df["macd"] < df["macd_signal"]) * 0.3
    # EMA momentum
    if "ema_diff" in df.columns:
        sig += (df["ema_diff"] > 0) * 0.3
        sig -= (df["ema_diff"] < 0) * 0.3
    # Bollinger squeeze & breakout
    if "bollinger_upper" in df.columns and "bollinger_lower" in df.columns and "Close" in df.columns:
        width = df["bollinger_upper"] - df["bollinger_lower"]
        squeeze = width < width.rolling(20).mean() * 0.8
        breakout = (df["Close"] > df["bollinger_upper"]) | (df["Close"] < df["bollinger_lower"])
        sig += (squeeze & breakout) * 0.5
    return sig.clip(-1, 1)

def generate_ml_signal(df, model=None):
    # Return predicted probability of positive return for each row
    if model is None:
        model = load_model()
    feats = [extract_feature_array(df.iloc[i]) for i in range(len(df))]
    if hasattr(model, "predict_proba"):
        preds = model.predict_proba(feats)[:,1]
    else:
        preds = model.predict(feats)
    return pd.Series(preds, index=df.index)

def smooth_signal(signal, smoothing_window=5):
    return signal.rolling(window=smoothing_window).mean().fillna(0)

def adaptive_threshold(df, target_profit=0.01):
    """
    Adaptive threshold selection based on expected value.
    """
    sig = smooth_signal(generate_signal(df))
    best_t, best_ev = 0.5, -float("inf")
    prices = df.get("Close") if "Close" in df.columns else pd.Series(0, index=df.index)
    for t in np.arange(0.05, 1.0, 0.05):
        combined_df = run_backtest(sig, prices, threshold=t)
        if "type" in combined_df.columns and combined_df.iloc[0].get("type") == "summary":
            trades_df = combined_df[combined_df["type"] == "trade"]
        else:
            trades_df = combined_df
        if trades_df.empty or "return" not in trades_df.columns:
            continue
        win_pct = (trades_df["return"] > 0).mean()
        avg_return = trades_df["return"].mean()
        ev = avg_return * win_pct
        if ev > best_ev:
            best_ev, best_t = ev, t
    return best_t

def track_trade_result(result, pair, action):
    if not result.get("filled", False):
        logging.warning(f"Trade for {pair} ({action}) was not filled.")
        return
    logging.info(
        f"Trade for {pair} ({action}) filled: "
        f"Order ID: {result.get('order_id')}, Amount: {result.get('amount')}, "
        f"Price: {result.get('order_price')}"
    )
