import joblib
import numpy as np
import logging

# --- Load Model ---
def load_model(filepath="state/meta_model_A.pkl"):
    try:
        model = joblib.load(filepath)
        return model
    except Exception as e:
        logging.getLogger(__name__).warning(f"[WARN] Could not load ML model: {e}")
        return None

# --- ML Confidence Function ---
def ml_confidence(df, model):
    logger = logging.getLogger(__name__)

    if model is None:
        logger.warning("⚠ No model loaded. Returning confidence 0.0")
        return 0.0

    features = ["rsi_z", "macd_z", "ema_diff_z", "volatility_z"]

    # --- Verify that required features exist ---
    missing = [f for f in features if f not in df.columns]
    if missing:
        logger.warning(f"❗ Missing features in df: {missing}")
        return 0.0

    X = df[features].dropna()

    if X.empty:
        logger.warning("❗ Feature DataFrame X is empty after dropna()")
        return 0.0

    # --- Debugging ---
    logger.info(f"🔍 DF columns before ML confidence: {list(df.columns)}")
    logger.info(f"🔍 Feature values preview:\n{X.tail()}")

    # --- Check model expected features ---
    if hasattr(model, "feature_names_in_"):
        logger.info(f"✅ Model expects features: {list(model.feature_names_in_)}")

    try:
        probs = model.predict_proba(X)
        confidences = probs[:, 1]  # Class 1 = positive signal
        return float(np.mean(confidences))
    except Exception as e:
        logger.warning(f"[WARN] ml_confidence fallback: {e}")
        return 0.0

# --- Train and Save Model ---
def train_and_save_model(df, variant="A"):
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    import joblib

    feature_cols = ['rsi_z', 'macd_z', 'ema_diff_z', 'volatility_z']
    target_col = 'indicator'

    df = df.dropna()
    X = df[feature_cols]
    y = df[target_col]

    if len(X) < 100:
        print("❌ Not enough data to train model.")
        return

    print(f"🔧 Training model on {len(X)} samples...")

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    print("✅ Model trained. Saving...")

    output_file = f"state/meta_model_{variant}.pkl"
    joblib.dump(model, output_file)

    print(f"✅ Model saved as {output_file}")