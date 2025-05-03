import os
import glob
import logging
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

# Configure logging to output to console with timestamp and level (if not already configured)
logger = logging.getLogger(__name__)
if not logger.handlers:  # avoid duplicate handlers if re-run
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Define the path for saving the model
MODEL_FILE_PATH = "state/meta_model_A.pkl"

def load_historical_data(data_dir="data/historical"):
    """
    Load and concatenate all CSV files from the specified directory.
    Returns a pandas DataFrame containing the combined data.
    """
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        logger.error(f"No CSV files found in {data_dir}.")
        return None
    df_list = []
    for file in files:
        try:
            df_part = pd.read_csv(file)
        except Exception as e:
            logger.error(f"Error reading {file}: {e}")
            return None
        df_list.append(df_part)
    df = pd.concat(df_list, ignore_index=True)
    logger.info(f"Loaded {len(df)} rows from {len(files)} historical data file(s).")
    return df

def prepare_features(df):
    """
    Given a DataFrame of historical OHLC data, compute technical features and return a new DataFrame ready for modeling.
    Requires 'Close' in df. Uses 'High' and 'Low' if available for ATR/volatility.
    """
    # Drop rows with missing essential columns
    if 'High' in df.columns and 'Low' in df.columns:
        df = df.dropna(subset=['Close', 'High', 'Low']).copy()
    else:
        df = df.dropna(subset=['Close']).copy()
    df.drop_duplicates(inplace=True)
    # Sort by time if a time column exists
    time_cols = [col for col in df.columns if "time" in col.lower() or "date" in col.lower()]
    if time_cols:
        df.sort_values(by=time_cols[0], inplace=True)
        df.reset_index(drop=True, inplace=True)

    # Compute RSI (14-period)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    period = 14
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Compute MACD (fast EMA 12, slow EMA 26)
    fast_span, slow_span = 12, 26
    ema_fast = df['Close'].ewm(span=fast_span, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow_span, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow

    # Compute EMA difference (Close minus 50-period EMA)
    ema_long = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_diff'] = df['Close'] - ema_long

    # Compute volatility (ATR % or rolling std dev of returns)
    if 'High' in df.columns and 'Low' in df.columns:
        atr_period = 14
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=True)
        df['ATR'] = tr.rolling(atr_period).mean()
        df['volatility'] = df['ATR'] / df['Close']
    else:
        df['volatility'] = df['Close'].pct_change().rolling(14).std()

    # Drop initial rows with NaNs from indicators
    df.dropna(inplace=True)
    return df

def add_labels(df, forward_horizon=5):
    """
    Add a binary 'Label' column to DataFrame based on forward returns over the given horizon.
    Label = 1 if price rises over the horizon, else 0.
    """
    df['future_close'] = df['Close'].shift(-forward_horizon)
    df['forward_return'] = (df['future_close'] - df['Close']) / df['Close']
    df['Label'] = (df['forward_return'] > 0).astype(int)
    df.dropna(subset=['Label'], inplace=True)
    return df

def train_model(df):
    """
    Train a RandomForestClassifier on the given DataFrame (with features and Label already prepared).
    Returns the trained model.
    """
    # Define feature columns to use for training (z-score normalized features)
    feature_cols = ['RSI', 'MACD', 'EMA_diff', 'volatility']
    # Compute z-scores for each feature
    feature_stats = {}
    for col in feature_cols:
        mean_val = df[col].mean()
        std_val = df[col].std()
        feature_stats[col] = (mean_val, std_val)
        if std_val is None or std_val == 0:
            df[f"{col}_zscore"] = 0.0
        else:
            df[f"{col}_zscore"] = (df[col] - mean_val) / std_val

    X_cols = [f"{col}_zscore" for col in feature_cols]
    X = df[X_cols]
    y = df['Label'].astype(int)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    # Attach feature stats and names for later use (e.g., ml_confidence)
    model.feature_stats = feature_stats
    model.feature_names = feature_cols
    model.feature_names_final = X_cols
    return model

def save_model(model, path):
    """
    Save the model to the given file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}.")

if __name__ == "__main__":
    # Continuous learning process
    df = load_historical_data("data/historical")
    if df is None or df.empty:
        logger.error("No data available for training. Exiting.")
        exit(1)
    # Prepare features
    df_feat = prepare_features(df)
    if df_feat.empty:
        logger.error("Feature preparation failed or resulted in no data. Exiting.")
        exit(1)
    # Add labels
    df_labeled = add_labels(df_feat, forward_horizon=5)
    if df_labeled.empty or 'Label' not in df_labeled.columns:
        logger.error("Label generation failed (insufficient future data). Exiting.")
        exit(1)
    # Prepare final dataset for training
    num_rows = len(df_labeled)
    # Train model
    model = train_model(df_labeled)
    logger.info(f"Model trained on {num_rows} samples with features {model.feature_names_final}.")
    # Save model to disk
    try:
        save_model(model, MODEL_FILE_PATH)
        logger.info("Continuous learning training completed successfully.")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        exit(1)
