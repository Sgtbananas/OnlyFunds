import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

MODEL_PATH = "state/ml_model.pkl"
META_MODEL_PATH = "state/meta_model_A.pkl"
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
    features = []
    labels = []
    for _, row in df.iterrows():
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

def extract_meta_features_labels(df):
    """
    For meta-learner: features = [regime_is_trending, s_trend, s_rev]
    labels = (return_pct > 0).astype(int)
    """
    try:
        from core.core_signals import classify_regime, signal_trending, signal_reversion
    except ImportError:
        return np.empty((0, 3)), np.empty((0,))
    df = df.copy()
    if "Close" not in df.columns:
        return np.empty((0, 3)), np.empty((0,))
    df = classify_regime(df)
    s_trend = signal_trending(df)
    s_rev = signal_reversion(df)
    regime_is_trending = (df['regime'] == 'trending').astype(int)
    features = np.stack([regime_is_trending, s_trend, s_rev], axis=1)
    labels = (df['return_pct'] > 0).astype(int).values if "return_pct" in df else np.zeros(len(df))
    return features, labels

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
    # Optionally print metrics for diagnostics
    try:
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
        print(f"ML Filter Training Accuracy: {acc:.3f} | ROC AUC: {auc:.3f}")
    except Exception:
        pass
    return True, f"ML filter trained and saved to {MODEL_PATH}"

def train_and_save_meta_model(meta_model_path=META_MODEL_PATH):
    df = load_trade_log()
    if df.empty:
        return False, "No trade history to train meta-learner."
    X, y = extract_meta_features_labels(df)
    if X.shape[0] == 0:
        return False, "No suitable features for meta-learner."
    model = LogisticRegression()
    model.fit(X, y)
    joblib.dump(model, meta_model_path)
    try:
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
        print(f"Meta-Learner Training Accuracy: {acc:.3f} | ROC AUC: {auc:.3f}")
    except Exception:
        pass
    return True, f"Meta-learner trained and saved to {meta_model_path}"

def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

def load_meta_model(path=META_MODEL_PATH):
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

def meta_learner_decision(meta_feats, model=None):
    """
    meta_feats: [regime_is_trending, s_trend, s_rev]
    Returns 1 for trade, 0 for no-trade.
    """
    if model is None:
        model = load_meta_model()
    if model is None:
        raise ValueError("Meta-learner model not trained/found.")
    # meta_feats should be shape (n_features,)
    out = model.predict([meta_feats])[0]
    return out

def meta_learner_proba(meta_feats, model=None):
    """
    Returns probability of positive trade outcome from meta-learner.
    """
    if model is None:
        model = load_meta_model()
    if model is None:
        raise ValueError("Meta-learner model not trained/found.")
    proba = model.predict_proba([meta_feats])[0, 1]
    return proba

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

def evaluate_meta_model(meta_model_path=META_MODEL_PATH):
    df = load_trade_log()
    if df.empty:
        print("No trade history for meta-learner evaluation.")
        return
    X, y = extract_meta_features_labels(df)
    if X.shape[0] == 0:
        print("No suitable features for meta-learner evaluation.")
        return
    model = joblib.load(meta_model_path)
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
    print("Meta-Learner Classification Report:")
    print(classification_report(y, y_pred))
    print(f"Accuracy: {acc:.3f} | ROC AUC: {auc:.3f}")