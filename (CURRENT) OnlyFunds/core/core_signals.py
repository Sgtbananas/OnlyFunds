import pandas as pd
import numpy as np
import logging
from core.backtester import run_backtest

COMMON_QUOTES = ["USDT", "BTC", "ETH", "BNB"]

def classify_regime(df, vol_window=20, mom_window=20, vol_thresh=0.01, mom_thresh=0.005):
    """
    Classify each row as 'trending' or 'reversion' regime.
    Trending = high volatility and strong momentum.
    """
    df = df.copy()
    df['volatility'] = df['Close'].pct_change().rolling(vol_window).std()
    df['momentum'] = df['Close'].pct_change(mom_window)
    df['regime'] = np.where(
        (df['volatility'] > vol_thresh) & (df['momentum'].abs() > mom_thresh),
        'trending', 'reversion'
    )
    return df

def signal_trending(df):
    """Momentum specialist: Simple EMA 20/50 crossover."""
    ema_fast = df['Close'].ewm(span=20).mean()
    ema_slow = df['Close'].ewm(span=50).mean()
    signal = (ema_fast > ema_slow).astype(float)
    # Optionally: scale by strength of crossover
    signal *= (ema_fast - ema_slow).abs() / (df['Close'] + 1e-8)
    return signal

def signal_reversion(df):
    """Mean reversion specialist: RSI, MACD, price distance to EMA."""
    signal = pd.Series(0.0, index=df.index)
    if "rsi" in df.columns:
        signal += ((df["rsi"] < 30) * 1.0)
        signal -= ((df["rsi"] > 70) * 1.0)
    if "macd" in df.columns:
        signal += ((df["macd"] < 0) * 0.5)
        signal -= ((df["macd"] > 0) * 0.5)
    if "ema_diff" in df.columns:
        signal -= (df["ema_diff"] > 0) * 0.3
        signal += (df["ema_diff"] < 0) * 0.3
    return signal

def generate_ensemble_signal(df, meta_model=None):
    """
    Regime classifier → specialist models → meta-learner (optional)
    Returns pd.Series of signal strength (clipped -1..1)
    """
    df = classify_regime(df)
    s_trend = signal_trending(df)
    s_rev = signal_reversion(df)
    if meta_model is not None:
        regime_is_trending = (df['regime'] == 'trending').astype(int)
        feats = np.stack([regime_is_trending, s_trend, s_rev], axis=1)
        meta_preds = meta_model.predict(feats)
        signal = pd.Series(meta_preds * 2 - 1, index=df.index)
        return signal.clip(-1, 1)
    else:
        signal = np.where(df['regime'] == 'trending', s_trend, s_rev)
        return pd.Series(signal, index=df.index).clip(-1, 1)

def generate_signal(df):
    """
    Backward-compatible: Use ensemble pipeline.
    """
    return generate_ensemble_signal(df, meta_model=None)

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

# For compatibility with ML filter code that may import this
def extract_feature_array(row):
    """
    Given a row, return np.array([rsi, macd, ema_diff, volatility]).
    """
    return np.array([
        row.get("rsi", 0),
        row.get("macd", 0),
        row.get("ema_diff", 0),
        row.get("volatility", 0)
    ])