import os
import json
import logging
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict
from datetime import datetime

# If fetch_klines and add_indicators are needed for training data:
from core.core_data import fetch_klines, add_indicators

# Path for the ML model file
MODEL_PATH = "state/ml_model.pkl"
# Default trading interval (used for fetching historical data)
DEFAULT_INTERVAL = "5m"

logger = logging.getLogger(__name__)

def load_model():
    """Load the trained ML model from disk, or return None if not available."""
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        logger.error(f"Failed to load ML model: {e}")
        return None

def train_and_save_model():
    """Train a RandomForest model on past trade outcomes and save it to disk."""
    trade_log_file = "state/trade_log.json"
    # Load trade history
    try:
        with open(trade_log_file, "r") as f:
            trade_log = json.load(f)
    except Exception as e:
        logger.error(f"Error reading trade log: {e}")
        return False, "Failed to load trade log"
    if not trade_log or len(trade_log) < 1:
        return False, "No trade data available for training"
    
    # Prepare training samples (features and outcome) from completed trades
    open_positions = {}
    samples = []  # Each sample: {"pair": ..., "open_time": ..., "outcome": ...}
    for entry in trade_log:
        side = entry.get("side")
        pair = entry.get("pair")
        if side == "BUY":
            # Record open trade info
            open_positions[pair] = entry
        elif side == "SELL" and pair in open_positions:
            # Trade closed; determine outcome
            open_entry = open_positions.pop(pair)
            result = entry.get("result")
            outcome = 0 if result == "SL" else 1  # 1 = successful (TP/TRAIL), 0 = stopped out (SL)
            samples.append({
                "pair": pair,
                "open_time": open_entry.get("timestamp"),
                "outcome": outcome
            })
    if not samples:
        return False, "No completed trades to train on"
    if len(samples) < 2:
        return False, f"Not enough training samples ({len(samples)})"
    
    # Group samples by trading pair for efficient feature extraction
    samples_by_pair = defaultdict(list)
    for sample in samples:
        try:
            open_dt = datetime.fromisoformat(sample["open_time"])
        except Exception as e:
            logger.warning(f"Skipping sample with invalid time {sample}: {e}")
            continue
        samples_by_pair[sample["pair"]].append((open_dt, sample["outcome"]))
    # Sort events by time for each pair
    for pair in samples_by_pair:
        samples_by_pair[pair].sort(key=lambda x: x[0])
    
    feature_matrix = []
    outcomes = []
    feature_names = None
    # Extract indicator features at each trade entry time
    for pair, events in samples_by_pair.items():
        # Fetch historical data for this pair (up to the last trade entry time)
        last_time = events[-1][0]
        try:
            df = fetch_klines(pair, interval=DEFAULT_INTERVAL, limit=1000)
        except Exception as e:
            logger.error(f"Data fetch failed for {pair}: {e}")
            continue
        if df is None or df.empty:
            logger.error(f"No data for {pair}, skipping ML training sample extraction.")
            continue
        df = add_indicators(df)
        # Ensure data sorted by time
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()
        elif "timestamp" in df.columns:
            df = df.sort_values("timestamp")
        elif "Time" in df.columns:
            df = df.sort_values("Time")
        # Iterate through each trade event for this pair
        for open_dt, outcome in events:
            # Find the latest row at or before open_dt
            entry_features = None
            if isinstance(df.index, pd.DatetimeIndex):
                # If DataFrame index is datetime
                if open_dt < df.index[0]:
                    continue  # trade is earlier than data range
                # get index position of last timestamp <= open_dt
                idx_pos = df.index.get_indexer([open_dt], method='ffill')[0]
                if idx_pos == -1:
                    continue
                entry_features = df.iloc[idx_pos]
            else:
                # Use a time column if present
                time_col = None
                for col in ["timestamp", "Time", "Datetime", "Date"]:
                    if col in df.columns:
                        time_col = col
                        break
                if not time_col:
                    logger.error(f"No timestamp column found in data for {pair}")
                    continue
                # Filter data up to open_dt
                df_times = pd.to_datetime(df[time_col])
                df_before = df[df_times <= open_dt]
                if df_before.empty:
                    continue
                entry_features = df_before.iloc[-1]
            if entry_features is None:
                continue
            # Determine feature columns (exclude raw OHLCV and timestamp columns)
            base_cols = {"Open", "High", "Low", "Close", "Volume", "Adj Close", 
                         "timestamp", "Time", "Datetime", "Date"}
            if feature_names is None:
                feature_names = [col for col in entry_features.index if col not in base_cols]
            # Gather feature values in the same order as feature_names
            try:
                features = [entry_features[col] for col in feature_names]
            except KeyError as e:
                logger.error(f"Missing expected feature {e} for {pair} at {open_dt}")
                continue
            feature_matrix.append(features)
            outcomes.append(outcome)
    if not feature_matrix:
        return False, "No training samples after feature extraction"
    
    X = np.array(feature_matrix, dtype=float)
    y = np.array(outcomes, dtype=int)
    if X.size == 0 or y.size == 0:
        return False, "Insufficient data for model training"
    # Compute feature mean and std for normalization
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=0)
    # Remove any constant features (std = 0)
    non_const_mask = (stds != 0)
    if not non_const_mask.all():
        X = X[:, non_const_mask]
        means = means[non_const_mask]
        stds = stds[non_const_mask]
        if feature_names:
            feature_names = [name for name, keep in zip(feature_names, non_const_mask) if keep]
        if X.shape[1] == 0:
            return False, "All features have zero variance"
    # Normalize features to z-scores
    X_norm = (X - means) / stds
    # Train the RandomForest classifier
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_norm, y)
    except Exception as e:
        logger.error(f"Model training error: {e}")
        return False, f"Training failed: {e}"
    # Attach feature metadata to the model for later use
    model.feature_names = feature_names or []
    model.feature_means = means
    model.feature_stds = stds
    # Save the trained model
    try:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)
    except Exception as e:
        logger.error(f"Failed to save ML model: {e}")
        return False, f"Model save failed: {e}"
    logger.info(f"Trained ML model on {len(y)} samples, features: {model.feature_names}")
    return True, f"Model trained on {len(y)} samples"

def ml_confidence(data):
    """Compute the model confidence (probability of positive outcome) for given data (DataFrame or Series)."""
    model = load_model()
    if model is None or not hasattr(model, "feature_names"):
        # If no model available, skip filtering (return high confidence)
        return 1.0
    # Select the latest data row and required features
    if isinstance(data, pd.DataFrame):
        if data.empty:
            return None
        try:
            # Use only the features the model was trained on
            row = data.iloc[-1][model.feature_names]
        except KeyError:
            # If indicators not present, compute them then try again
            data = add_indicators(data.copy())
            try:
                row = data.iloc[-1][model.feature_names]
            except Exception as e:
                logger.error(f"ml_confidence: required features missing from data: {e}")
                return None
    elif isinstance(data, pd.Series):
        row = data
        if not set(model.feature_names).issubset(row.index):
            logger.error("ml_confidence: input Series is missing required features")
            return None
        row = row[model.feature_names]
    else:
        # Try to convert other data formats (dict or list) to a Series
        try:
            row = pd.Series(data)
            row = row[model.feature_names]
        except Exception as e:
            logger.error(f"ml_confidence: unsupported data format: {e}")
            return None
    # Normalize the features using model's stored mean and std
    vals = np.array(row.values, dtype=float)
    means = np.array(model.feature_means, dtype=float)
    stds = np.array(model.feature_stds, dtype=float)
    stds[stds == 0] = 1e-9  # safeguard against division by zero
    z = (vals - means) / stds
    # Predict probability of positive class (outcome=1)
    try:
        proba = model.predict_proba(z.reshape(1, -1))[0]
    except Exception as e:
        logger.error(f"ml_confidence: model prediction failed: {e}")
        return None
    if len(model.classes_) == 2:
        # Determine index of class '1'
        try:
            class_one_index = list(model.classes_).index(1)
        except ValueError:
            class_one_index = 1  # if class 1 not found, default to second class
        confidence = proba[class_one_index]
    else:
        # For non-binary, use the maximum probability as a conservative measure
        confidence = proba.max()
    return float(confidence)
