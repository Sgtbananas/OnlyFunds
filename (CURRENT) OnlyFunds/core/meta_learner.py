import numpy as np
import pandas as pd
import joblib
import os

META_MODEL_PATH = "state/meta_model_A.pkl"

def load_meta_model(path=META_MODEL_PATH):
    if os.path.exists(path):
        return joblib.load(path)
    return None

def select_strategy(performance_dict, meta_model=None):
    """
    performance_dict: {strategy: {'win_rate':..., 'sharpe':..., ...}, ...}
    meta_model: sklearn-like model or None
    Returns: selected strategy name
    """
    strategies = list(performance_dict.keys())
    features = []
    for strat in strategies:
        perf = performance_dict[strat]
        # Features: win_rate, sharpe, drawdown, total_pnl
        features.append([
            perf.get('win_rate', 0),
            perf.get('sharpe_ratio', 0),
            perf.get('max_drawdown', 0),
            perf.get('total_pnl', 0),
        ])
    features = np.array(features)
    if meta_model is not None:
        preds = meta_model.predict(features)
        idx = np.argmax(preds)  # Pick highest scoring
    else:
        idx = np.argmax([f[1] for f in features])  # Use Sharpe as tiebreaker fallback
    return strategies[idx]