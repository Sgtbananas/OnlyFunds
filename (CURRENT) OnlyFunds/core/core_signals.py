import pandas as pd
import numpy as np
import logging

COMMON_QUOTES = ["USDT", "BTC", "ETH", "BNB"]

def classify_regime(df, vol_window=20, mom_window=20, vol_thresh=0.005, mom_thresh=0.002):
    """Classify each row as 'trending' or 'reversion'."""
    df = df.copy()
    df['volatility'] = df['Close'].pct_change().rolling(vol_window).std()
    df['momentum'] = df['Close'].pct_change(mom_window)
    df['regime'] = np.where(
        (df['volatility'] > vol_thresh) & (df['momentum'].abs() > mom_thresh),
        'trending', 'reversion'
    )
    return df

def signal_trending(df, ema_fast_window=20, ema_slow_window=50):
    """Momentum specialist."""
    ema_fast = df['Close'].ewm(span=ema_fast_window).mean()
    ema_slow = df['Close'].ewm(span=ema_slow_window).mean()
    signal = (ema_fast > ema_slow).astype(float)
    signal *= (ema_fast - ema_slow).abs() / (df['Close'] + 1e-8)
    return signal.clip(0, 1)  # Always non-negative

def signal_reversion(df, rsi_low=35, rsi_high=65):
    """Mean reversion specialist."""
    signal = pd.Series(0.0, index=df.index)
    if "rsi" in df.columns:
        signal += ((df["rsi"] < rsi_low) * 1.0)
        signal -= ((df["rsi"] > rsi_high) * 1.0)
    if "macd" in df.columns:
        signal += ((df["macd"] < 0) * 0.5)
        signal -= ((df["macd"] > 0) * 0.5)
    if "ema_diff" in df.columns:
        signal -= (df["ema_diff"] > 0) * 0.3
        signal += (df["ema_diff"] < 0) * 0.3
    return signal.clip(-1, 1)

def generate_ensemble_signal(
    df, meta_model=None,
    regime_kwargs=None,
    trend_kwargs=None,
    reversion_kwargs=None
):
    """Regime classifier → specialist signals → combine."""
    regime_kwargs = regime_kwargs or {}
    trend_kwargs = trend_kwargs or {}
    reversion_kwargs = reversion_kwargs or {}

    df = classify_regime(df, **regime_kwargs)
    s_trend = signal_trending(df, **trend_kwargs)
    s_rev = signal_reversion(df, **reversion_kwargs)

    if meta_model is not None:
        regime_is_trending = (df['regime'] == 'trending').astype(int)
        feats = np.stack([regime_is_trending, s_trend, s_rev], axis=1)
        meta_preds = meta_model.predict(feats)
        signal = pd.Series(meta_preds * 2 - 1, index=df.index)
        return signal.clip(-1, 1)
    else:
        signal = np.where(df['regime'] == 'trending', s_trend, s_rev)
        return pd.Series(signal, index=df.index).clip(-1, 1)

def generate_signal(df, indicator_params=None):
    """Pipeline for generating the final signal."""
    indicator_params = indicator_params or {}
    sig = generate_ensemble_signal(
        df,
        meta_model=None,
        regime_kwargs=indicator_params.get("regime_kwargs", {}),
        trend_kwargs=indicator_params.get("trend_kwargs", {}),
        reversion_kwargs=indicator_params.get("reversion_kwargs", {}),
    )
    return sig.clip(-1, 1)

def smooth_signal(signal, smoothing_window=3):
    """Smooth signals to reduce noise."""
    return signal.rolling(window=smoothing_window).mean().fillna(0)
