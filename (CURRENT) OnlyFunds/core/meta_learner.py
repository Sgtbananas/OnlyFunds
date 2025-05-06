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
    """Select the best strategy based on the meta-model or fallback."""
    # If no performance data, return default strategy
    if not performance_dict or not isinstance(performance_dict, dict) or len(performance_dict) == 0:
        return "default_strategy", None

    # List of strategies to choose from
    strategies = list(performance_dict.keys())

    if not strategies:
        return "default_strategy", None

    # Prepare feature data for each strategy
    features = []
    for strat in strategies:
        perf = performance_dict.get(strat, {})
        features.append([
            perf.get('win_rate', 0),
            perf.get('sharpe_ratio', 0),
            perf.get('max_drawdown', 0),
            perf.get('total_pnl', 0),
            perf.get('volatility', 0),
            perf.get('trade_count', 0),
        ])

    features = np.array(features)

    if meta_model is not None and features.shape[0] > 0:
        try:
            preds = meta_model.predict(features)
            confidences = meta_model.predict_proba(features)
            selected_idx = int(np.argmax(confidences))
            return strategies[selected_idx], confidences[selected_idx]
        except Exception as e:
            print(f"[meta_learner] Model prediction failed: {e}. Defaulting to first strategy.")
            return strategies[0], None

    return strategies[0], None
