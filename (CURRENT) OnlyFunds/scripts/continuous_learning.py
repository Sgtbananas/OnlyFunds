import pandas as pd
from core.ml_filter import train_and_save_model
from data.historical.historical import load_historical_data
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def continuous_learning_loop():
    logger.info("🔄 Starting continuous learning loop...")

    data = load_historical_data()

    if data is None or data.empty:
        logger.warning("❗ No historical data available for retraining.")
        return

    logger.info(f"✅ Loaded {len(data)} rows of historical data.")

    # --- Check required features ---
    required = ["rsi_z", "macd_z", "ema_diff_z", "volatility_z", "indicator"]
    missing = [f for f in required if f not in data.columns]
    if missing:
        logger.warning(f"❗ Missing features for training: {missing}")
        return

    logger.info("✅ All required features found. Beginning training...")

    # --- Train Model ---
    train_and_save_model(data, variant="A")
    logger.info("✅ Training complete and model saved as state/meta_model_A.pkl")

if __name__ == "__main__":
    continuous_learning_loop()
