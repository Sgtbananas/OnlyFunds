import os
import numpy as np
import pandas as pd
import joblib

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

def extract_feature_array(row):
    # Extend as needed for more features!
    return [
        row.get("rsi", 50),
        row.get("macd", 0),
        row.get("ema_diff", 0),
        row.get("volatility", 0)
    ]

def extract_features_labels(df):
    features = []
    labels = []
    for _, row in df.iterrows():
        # Return only if all features are present
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

def fit_best_ml_model(X, y, path=MODEL_PATH):
    # Try LightGBM, XGBoost, then LogisticRegression
    model = None
    msg = ""
    if X.shape[0] == 0:
        return None, "No suitable features for ML ensemble."
    try:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(n_estimators=100)
        model.fit(X, y)
        msg = "Trained LightGBM model."
    except Exception as e:
        try:
            import xgboost as xgb
            model = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="logloss")
            model.fit(X, y)
            msg = "Trained XGBoost model."
        except Exception as e2:
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression()
            model.fit(X, y)
            msg = "Trained LogisticRegression model."
    joblib.dump(model, path)
    return model, f"{msg} Saved to {path}"

def train_and_save_model():
    df = load_trade_log()
    if df.empty:
        return False, "No trade history to train ML filter."
    X, y = extract_features_labels(df)
    if X.shape[0] == 0:
        return False, "No suitable features in trade log to train ML filter."
    model, msg = fit_best_ml_model(X, y)
    return True, msg

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
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba([features])[0, 1]
    else:
        prob = float(model.predict([features])[0])
    return prob
