import os
import glob
import logging
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

# Initialize module logger
logger = logging.getLogger(__name__)

# Global cache for the loaded model to avoid repeated disk reads
_model = None

# Model file path (ensure this matches the main app expectation)
MODEL_FILE_PATH = "state/meta_model_A.pkl"

def load_model():
    """
    Load the trained ML model from disk (state/meta_model_A.pkl).
    Returns the model if available, otherwise None.
    """
    global _model
    if _model is not None:
        return _model
    try:
        _model = joblib.load(MODEL_FILE_PATH)
        logger.info(f"Loaded ML model from '{MODEL_FILE_PATH}'.")
    except FileNotFoundError:
        logger.warning(f"No ML model found at '{MODEL_FILE_PATH}'.")
        _model = None
    except Exception as e:
        logger.error(f"Error loading ML model: {e}")
        _model = None
    return _model

def ml_confidence(data):
    """
    Compute model confidence using precomputed z-scores.
    """
    model = load_model()
    if model is None:
        logger.error("No model loaded.")
        return 0.0

    # Check data
    if data is None or len(data) == 0:
        logger.error("No data provided to ml_confidence.")
        return 0.0

    if isinstance(data, pd.DataFrame):
        sample = data.iloc[-1]
    elif isinstance(data, pd.Series):
        sample = data
    else:
        logger.error("Unsupported data type.")
        return 0.0

    expected_cols = model.feature_names_final  # Should be ["rsi_zscore", "macd_zscore", "ema_diff_zscore", "volatility_zscore"]

    missing = [c for c in expected_cols if c not in sample.index]
    if missing:
        logger.error(f"Missing expected z-score columns: {missing}")
        return 0.0

    X = pd.DataFrame([sample[expected_cols].values], columns=expected_cols)

    try:
        proba = model.predict_proba(X)[0][1]
        return float(proba)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return 0.0

    # Ensure we have a pandas Series for the sample (for consistent indexing)
    sample = sample.squeeze()  # In case it's a one-row DataFrame, convert to Series

    # If the sample itself has multiple entries (unlikely for Series), take the last value for each (handled by squeeze above).
    # Prepare raw feature values from the sample
    required_features = ["RSI", "MACD", "EMA_diff", "volatility"]
    raw_vals = {}
    for feat in required_features:
        if feat in sample.index:
            raw_vals[feat] = sample[feat]
        else:
            raw_vals[feat] = None

    # Special case: if volatility not directly provided but ATR and Close are present, compute volatility = ATR/Close
    if raw_vals.get("volatility") is None:
        if "ATR" in sample.index and "Close" in sample.index:
            try:
                raw_vals["volatility"] = sample["ATR"] / sample["Close"]
                logger.debug("Computed volatility from ATR for ml_confidence.")
            except Exception:
                raw_vals["volatility"] = None

    # If EMA_diff not provided but an EMA value is present (e.g., EMA50), compute difference from Close
    if raw_vals.get("EMA_diff") is None:
        # Find any column name that looks like 'EMA'
        ema_cols = [col for col in sample.index if col.upper().startswith("EMA") and col not in ["EMA_diff"]]
        if ema_cols:
            try:
                ema_val = sample[ema_cols[0]]
                if "Close" in sample.index:
                    raw_vals["EMA_diff"] = sample["Close"] - ema_val
                    logger.debug(f"Computed EMA_diff as Close - {ema_cols[0]} for ml_confidence.")
            except Exception:
                raw_vals["EMA_diff"] = None

    # Fill missing features with their training mean (to produce a neutral z-score of 0)
    missing_feats = []
    feature_stats = getattr(model, "feature_stats", {})
    for feat in required_features:
        if raw_vals.get(feat) is None:
            # If we have stats for this feature from training, use mean; otherwise default to 0
            if feat in feature_stats:
                raw_vals[feat] = feature_stats[feat][0]  # training mean
            else:
                raw_vals[feat] = 0.0
            missing_feats.append(feat)
    if missing_feats:
        logger.warning(f"ml_confidence: input missing features {missing_feats}. Using training means for those.")

    # Compute z-scores for each feature based on training stats
    X_input = []
    for feat in required_features:
        # Use training mean and std from model.feature_stats
        if feat in feature_stats:
            mean_val, std_val = feature_stats[feat]
            if std_val is None or std_val == 0:
                z = 0.0
            else:
                z = (raw_vals[feat] - mean_val) / std_val
        else:
            # If feature_stats not available (should not happen if model is trained using this module)
            z = raw_vals[feat]
        X_input.append(z)
    X_input = np.array(X_input).reshape(1, -1)

    # Construct DataFrame with the same feature columns as training, to avoid feature name issues
    if hasattr(model, "feature_names_final"):
        feature_cols = model.feature_names_final
    elif hasattr(model, "feature_names_in_"):
        # Use feature names that the model was trained with
        feature_cols = list(model.feature_names_in_)
    else:
        # Fallback: create names by appending "_zscore" to raw feature names
        feature_cols = [f"{feat}_zscore" for feat in required_features]
    X_df = pd.DataFrame(X_input, columns=feature_cols)

    try:
        proba = model.predict_proba(X_df)[0][1]  # probability of class 1 (positive outcome)
    except Exception as e:
        logger.error(f"Error computing model confidence: {e}")
        return 0.0
    logger.debug(f"Computed ml_confidence probability: {proba:.4f}")
    return float(proba)

def train_and_save_model(data_files, model_file="state/meta_model_A.pkl"):
    """
    Train a RandomForestClassifier model on historical data and save it to disk.
    Args:
        data_files: List of CSV file paths containing historical price data.
        model_file: Path where the trained model will be saved (default: state/meta_model_A.pkl).
    Returns:
        Tuple (success: bool, message: str) indicating whether training was successful and a status message.
    """
    try:
        # Read and concatenate all historical data
        df_list = []
        for file in data_files:
            try:
                df_part = pd.read_csv(file)
            except Exception as e:
                logger.error(f"Failed to read {file}: {e}")
                return False, f"Failed to read {file}: {e}"
            df_list.append(df_part)
        df = pd.concat(df_list, ignore_index=True)
        raw_rows = len(df)
        logger.info(f"Loaded {raw_rows} rows of historical data from {len(data_files)} file(s).")

        # Basic cleaning: drop any rows missing essential price data
        if 'High' in df.columns and 'Low' in df.columns:
            df.dropna(subset=['Close', 'High', 'Low'], inplace=True)
        else:
            df.dropna(subset=['Close'], inplace=True)
        # Remove exact duplicate rows (if any)
        df.drop_duplicates(inplace=True)
        cleaned_rows = len(df)
        if cleaned_rows < raw_rows:
            logger.info(f"Cleaned data: {raw_rows - cleaned_rows} rows removed (NaN or duplicates). Remaining {cleaned_rows} rows.")

        # Ensure the data is sorted by time if a time column exists
        time_cols = [col for col in df.columns if "time" in col.lower() or "date" in col.lower()]
        if time_cols:
            df.sort_values(by=time_cols[0], inplace=True)
            df.reset_index(drop=True, inplace=True)

        # Feature Engineering
        # 1. Compute RSI (Relative Strength Index, 14-period by default)
        if 'Close' not in df.columns:
            logger.error("Close price column missing in historical data.")
            return False, "Missing Close price in data."
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        window = 14
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # 2. Compute MACD (Moving Average Convergence Divergence: fast EMA 12, slow EMA 26)
        fast_span, slow_span, signal_span = 12, 26, 9
        ema_fast = df['Close'].ewm(span=fast_span, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=slow_span, adjust=False).mean()
        df['MACD'] = ema_fast - ema_slow
        # (Optionally compute MACD signal and histogram if needed in future)
        # df['MACD_signal'] = df['MACD'].ewm(span=signal_span, adjust=False).mean()
        # df['MACD_hist'] = df['MACD'] - df['MACD_signal']

        # 3. Compute EMA difference (difference between price and a longer-term EMA, e.g. 50-period)
        ema_long_span = 50
        ema_long = df['Close'].ewm(span=ema_long_span, adjust=False).mean()
        df['EMA_diff'] = df['Close'] - ema_long

        # 4. Compute volatility measure (ATR-based or rolling std of returns)
        if 'High' in df.columns and 'Low' in df.columns:
            atr_window = 14
            high_low = df['High'] - df['Low']
            high_close = (df['High'] - df['Close'].shift()).abs()
            low_close = (df['Low'] - df['Close'].shift()).abs()
            # True range
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=True)
            df['ATR'] = tr.rolling(atr_window).mean()
            df['volatility'] = df['ATR'] / df['Close']
        else:
            # Fallback: use rolling standard deviation of close-to-close returns as volatility
            df['volatility'] = df['Close'].pct_change().rolling(14).std()

        # Drop initial rows where indicators couldn't be computed (NaN values)
        df.dropna(inplace=True)
        if len(df) == 0:
            logger.error("Historical data is insufficient after feature computation (all rows dropped).")
            return False, "Not enough data after feature computation."

        # Normalize features via z-score
        feature_cols = ['RSI', 'MACD', 'EMA_diff', 'volatility']
        feature_stats = {}
        for col in feature_cols:
            mean_val = df[col].mean()
            std_val = df[col].std()
            feature_stats[col] = (mean_val, std_val)
            if std_val is None or std_val == 0:
                # Avoid division by zero (if std is zero, data is constant; z-score would be 0)
                df[f"{col}_zscore"] = 0.0
            else:
                df[f"{col}_zscore"] = (df[col] - mean_val) / std_val

        # Label Generation: forward return over a horizon to classify outcome
        horizon = 5  # number of periods ahead to evaluate return (can adjust as needed)
        # Compute forward return as percentage change over the horizon
        df['future_close'] = df['Close'].shift(-horizon)
        df['forward_return'] = (df['future_close'] - df['Close']) / df['Close']
        # Define label: 1 if forward return > 0 (positive outcome), else 0
        df['Label'] = (df['forward_return'] > 0).astype(int)
        # Drop final rows where forward_return could not be calculated (NaN in future_close/forward_return)
        df.dropna(subset=['Label'], inplace=True)

        if len(df) < 1:
            logger.error("No training samples after label generation (check horizon and data length).")
            return False, "No data to train after labeling."

        # Prepare training data
        X_cols = [f"{col}_zscore" for col in feature_cols]
        X = df[X_cols]
        y = df['Label'].astype(int)
        num_samples = len(X)
        num_features = len(X_cols)
        pos_ratio = y.mean() if num_samples > 0 else 0

        logger.info(f"Training data prepared: {num_samples} samples, {num_features} features.")
        logger.info(f"Feature columns: {X_cols}")
        logger.info(f"Positive class proportion: {pos_ratio:.2%}")

        # Check for class balance or single-class issues
        if y.nunique() < 2:
            logger.warning("Training labels contain only one class. The model will not be predictive.")
        if num_samples < 10:
            logger.warning("Very few training samples available. Model performance may be poor or unreliable.")

        # Train the model (Random Forest Classifier)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        # Attach training stats to model for use in ml_confidence (for consistent feature scaling)
        model.feature_stats = feature_stats
        model.feature_names = feature_cols  # raw feature names
        model.feature_names_final = X_cols  # z-score feature names used in training

        # Save the trained model to disk
        os.makedirs(os.path.dirname(MODEL_FILE_PATH), exist_ok=True)
        try:
            joblib.dump(model, MODEL_FILE_PATH)
        except Exception as e:
            logger.error(f"Failed to save model to '{MODEL_FILE_PATH}': {e}")
            return False, f"Model training successful, but saving failed: {e}"

        # Update global model cache
        _model = model
        logger.info(f"Model trained and saved to '{MODEL_FILE_PATH}'.")
        return True, f"Trained model on {num_samples} samples with {num_features} features."
    except Exception as e:
        logger.exception("Exception during model training:")
        return False, f"Error during model training: {e}"
