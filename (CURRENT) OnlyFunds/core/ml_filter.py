import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

MODEL_PATH = "state/ml_model.pkl"
TRADE_LOG_PATH = "state/trade_log.json"

def load_trade_log(trade_log_path=TRADE_LOG_PATH):
    import json
    if not os.path.exists(trade_log_path):
        return pd.DataFrame()
    with open(trade_log_path, "r") as f:
        trades = json.load(f)
    df = pd.DataFrame(trades)
    return df

def extract_features_labels(df):
    # Example: Use basic numeric features; customize as needed
    features = []
    labels = []
    for _, row in df.iterrows():
        # Features: rsi, macd, ema_diff, volatility, etc. (extend as needed)
        if all(x in row for x in ["rsi", "macd", "ema_diff", "volatility", "return_pct"]):
            features.append([
                row["rsi"],
                row["macd"],
                row["ema_diff"],
                row["volatility"]
            ])
            labels.append(int(row["return_pct"] > 0))
    if not features:
        return np.empty((0, 4)), np.empty((0,))
    return np.array(features), np.array(labels)

def train_and_save_model():
    df = load_trade_log()
    if df.empty:
        return False, "No trade history to train ML filter."
    X, y = extract_features_labels(df)
    if X.shape[0] == 0:
        return False, "No suitable features in trade log to train ML filter."
    model = LogisticRegression()
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return True, f"ML filter trained and saved to {MODEL_PATH}"

def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

def ml_confidence(features, model=None):
    """
    features: [rsi, macd, ema_diff, volatility]
    Returns probability of positive return.
    """
    if model is None:
        model = load_model()
    if model is None:
        raise ValueError("ML model not trained/found.")
    prob = model.predict_proba([features])[0, 1]
    return prob
