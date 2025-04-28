# core/atr_tuner.py

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# where to store your trained ATR‐tuner model
MODEL_PATH = os.path.join("state", "atr_tuner.pkl")


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with at least columns ['Close','ATR','rsi'],
    compute rolling statistics and return a features DataFrame.
    """
    feats = pd.DataFrame({
        "atr_mean":    df["ATR"].rolling(100).mean(),
        "atr_std":     df["ATR"].rolling(100).std(),
        "volatility":  df["Close"].pct_change().rolling(20).std(),
        "rsi":         df["rsi"],
    })
    return feats.dropna()


def generate_training_data(
    labeled_dfs: list[tuple[pd.DataFrame, float, float, float]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build X (features) and y (targets) from historical examples.
    Each entry in labeled_dfs is (df, stop_mult, tp_mult, trail_mult).
    We take the last row of features for each df as its descriptor.
    """
    X_rows, y_stop, y_tp, y_trail = [], [], [], []
    for df, stop_m, tp_m, trail_m in labeled_dfs:
        feats = extract_features(df)
        if feats.empty:
            continue
        last = feats.iloc[-1]
        X_rows.append(last)
        y_stop.append(stop_m)
        y_tp.append(tp_m)
        y_trail.append(trail_m)

    X = pd.DataFrame(X_rows)
    y = pd.DataFrame({
        "stop_mult":  y_stop,
        "tp_mult":    y_tp,
        "trail_mult": y_trail
    })
    return X, y


def train_atr_tuner(
    labeled_dfs: list[tuple[pd.DataFrame, float, float, float]],
    model_path: str = MODEL_PATH,
) -> RandomForestRegressor:
    """
    Train a RandomForestRegressor on your labeled examples
    and persist it to disk.
    """
    X, y = generate_training_data(labeled_dfs)
    if X.empty:
        raise ValueError("No training data available (all feature windows empty).")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"[atr_tuner] R² on held‐out data: {score:.3f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"[atr_tuner] Model saved to {model_path}")
    return model


def load_atr_tuner(model_path: str = MODEL_PATH) -> RandomForestRegressor | None:
    """Load a previously trained ATR‐tuner model, or None if missing."""
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


def predict_atr_multipliers(
    df: pd.DataFrame,
    model: RandomForestRegressor | None = None
) -> tuple[float, float, float]:
    """
    Given your latest OHLCV+ATR df, return
      (stop_loss_atr_mult, take_profit_atr_mult, trail_atr_mult).

    Falls back to sensible defaults if no model is available.
    """
    if model is None:
        model = load_atr_tuner()

    feats = extract_features(df)
    if feats.empty:
        # no history to compute features on → fallback
        return 1.0, 2.0, 1.0

    last = feats.iloc[-1].values.reshape(1, -1)
    if model is not None:
        stop_m, tp_m, trail_m = model.predict(last)[0]
        return float(stop_m), float(tp_m), float(trail_m)
    else:
        return 1.0, 2.0, 1.0
