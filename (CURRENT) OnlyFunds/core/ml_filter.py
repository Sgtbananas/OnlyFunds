from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np
import logging
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_FEATURES = [
    "rsi", "macd", "macd_signal", "bollinger_mid", "bollinger_upper", 
    "bollinger_lower", "ema", "ema_diff", "volatility", "ATR",
    "rsi_z", "macd_z", "ema_diff_z", "volatility_z", "indicator"
]

def load_model(path):
    try:
        model = joblib.load(path)
        logger.info(f"✅ ML model loaded successfully from {path}")
        return model
    except Exception as e:
        logger.error(f"❌ Failed to load ML model from {path}: {e}")
        return None

def ml_confidence(df, model):
    if model is None:
        logger.warning("⚠ No ML model loaded. Returning 0.0 confidence.")
        return 0.0

    if df is None or df.empty:
        logger.warning("⚠ Dataframe is empty or None. Returning 0.0 confidence.")
        return 0.0

    missing = [f for f in REQUIRED_FEATURES if f not in df.columns]
    if missing:
        logger.warning(f"⚠ Missing required features in dataframe: {missing}")
        return 0.0

    X = df[REQUIRED_FEATURES].dropna()
    if X.empty:
        logger.warning("⚠ No valid rows after dropping NAs. Returning 0.0 confidence.")
        return 0.0

    try:
        logger.info(f"🔎 Feature values preview (last 5 rows):\n{X.tail()}")
        logger.info(f"🔎 Passing {len(X)} samples to ML model for confidence prediction.")
        if hasattr(model, "feature_names_in_"):
            model_features = list(model.feature_names_in_)
            logger.info(f"✅ Model expects features: {model_features}")

        probs = model.predict_proba(X)
        last_probs = probs[-1]
        confidence = last_probs[1] if len(last_probs) > 1 else last_probs[0]
        logger.info(f"✅ Model confidence: {confidence:.4f}")
        return confidence

    except Exception as e:
        logger.error(f"❌ Confidence prediction failed: {e}")
        return 0.0

def train_and_save_model():
    try:
        trade_log = load_json("state/trade_log.json", [])
        closed = [t for t in trade_log if t.get("result") in ("TP","SL","TRAIL")]
        if len(closed) < 20:
            return (False, "Not enough data to retrain model.")
    try:
        clf = RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            random_state=42
        )
        clf.fit(X, y)
        joblib.dump(clf, path)
        logger.info(f"✅ ML model trained and saved to {path}")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        model = LogisticRegression()
        model.fit(X, y)
        joblib.dump(model, META_MODEL_PATH)
        # Update in-memory model if needed:
        global META_MODEL
        META_MODEL = model
        return (True, f"Trained on {len(X)} trades.")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return (False, str(e))
