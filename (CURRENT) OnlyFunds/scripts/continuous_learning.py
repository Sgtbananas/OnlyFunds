import pandas as pd
from core.core_data import fetch_klines, add_indicators
from utils.helpers import load_json
import joblib
from sklearn.linear_model import LogisticRegression


from core.ml_filter import train_and_save_model
success, msg = train_and_save_model()
if success:
    print("✅ Meta-model retrained successfully:", msg)
else:
    print("⚠️ Meta-model retraining failed:", msg)


# Load trade history
trade_log = load_json("state/trade_log.json", default=[])
closed_trades = [t for t in trade_log if t.get("result") in ("TP", "SL", "TRAIL")]

if len(closed_trades) < 20:
    print("Not enough completed trades to train the model.")
    exit(0)

# Prepare training examples
training_rows = []
for trade in closed_trades:
    pair = trade["pair"]; result = trade["result"]; entry_time = trade["timestamp"]
    label = 1 if result in ("TP", "TRAIL") else 0

    # Fetch historical data up to the entry time
    df = fetch_klines(pair, interval=DEFAULT_INTERVAL, end_time=entry_time, limit=1000)
    if df is None or df.empty: 
        continue
    df = add_indicators(df)
    # Get features at the last available time before entry (the entry candle)
    features = df.iloc[-1].to_dict()
    features["label"] = label
    training_rows.append(features)

# Create DataFrame
df_train = pd.DataFrame(training_rows)
X = df_train.drop(columns=["label"])
y = df_train["label"]

# Drop any non-indicator columns if present (e.g., raw OHLCV data not needed for model)
for col in ["Open","High","Low","Close","Volume","Timestamp"]:
    if col in X.columns:
        X.drop(columns=col, inplace=True)

# Train the classifier
model = LogisticRegression()
model.fit(X, y)

# Save model to disk
joblib.dump(model, "state/meta_model_A.pkl")
print(f"✅ Trained meta-model on {len(X)} samples. Saved to state/meta_model_A.pkl.")
def ml_confidence(df: pd.DataFrame) -> float:
    """
    Use the meta-model to compute a confidence score (probability) for the given data.
    Logs the expected vs actual feature columns for transparency.
    """
    global META_MODEL
    if META_MODEL is None:
        logger.info("ML model not loaded; cannot compute confidence.")
        return None
    try:
        # Determine expected feature names from the model
        if hasattr(META_MODEL, "feature_names_in_"):
            expected_features = list(META_MODEL.feature_names_in_)
        else:
            # Fallback: use the number of features the model was trained on
            expected_features = []
            if hasattr(META_MODEL, "n_features_in_"):
                exp_count = META_MODEL.n_features_in_
                expected_features = list(df.columns[:exp_count])
        actual_features = list(df.columns)
        logger.info(f"ML Confidence: Model expects features {expected_features}; Received DataFrame columns {actual_features}")
        
        # Ensure DataFrame has the expected columns
        X = df[expected_features]
        # For safety, use the latest row of X (if df has multiple rows)
        if len(X) > 1:
            X = X.tail(1)
        # Get confidence score (probability of class 1 if available)
        if hasattr(META_MODEL, "predict_proba"):
            prob = META_MODEL.predict_proba(X)[0][1]  # probability of positive class
        else:
            # If model has no predict_proba (e.g., a custom model), use predict
            prob = float(META_MODEL.predict(X)[0])
        return prob
    except Exception as e:
        logger.error(f"ml_confidence error: {e}")
        return None

