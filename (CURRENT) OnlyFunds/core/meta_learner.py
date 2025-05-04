import numpy as np
import pandas as pd
import joblib
import os

# Path to the meta-model
META_MODEL_PATH = "state/meta_model_A.pkl"

def load_meta_model(path=META_MODEL_PATH):
    """Load the meta-model if it exists."""
    if os.path.exists(path):
        return joblib.load(path)
    return None

def select_strategy(performance_dict, meta_model=None):
    """Select the best strategy based on the meta-model."""
    # List of strategies to choose from
    strategies = list(performance_dict.keys())
    
    # Prepare feature data for each strategy
    features = []
    for strat in strategies:
        perf = performance_dict[strat]
        features.append([
            perf.get('win_rate', 0),         # Strategy's win rate
            perf.get('sharpe_ratio', 0),     # Sharpe ratio
            perf.get('max_drawdown', 0),     # Maximum drawdown
            perf.get('total_pnl', 0),        # Total PnL
            perf.get('volatility', 0),       # Volatility
            perf.get('trade_count', 0),      # Number of trades
        ])
    
    # Convert features list into a numpy array
    features = np.array(features)
    
    # If a model exists, use it to predict the best strategy
    if meta_model is not None:
        preds = meta_model.predict(features)  # Get predicted strategy scores
        confidences = meta_model.predict_proba(features)  # Get prediction probabilities (confidence levels)
        selected_idx = np.argmax(confidences)  # Select the strategy with the highest confidence
        return strategies[selected_idx], confidences[selected_idx]
    
    # Default to the first strategy if no model is provided
    return strategies[0], None  # No AI/ML model, so select the first strategy by default
