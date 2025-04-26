from sklearn.linear_model import LogisticRegression
import pickle

def train_ml_filter(X, y, model_path="ml_model.pkl"):
    model = LogisticRegression()
    model.fit(X, y)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

def load_model(path="ml_model.pkl"):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

def ml_confidence(model, features):
    """features: [rsi, macd, ema_diff, volatility, ...]"""
    prob = model.predict_proba([features])[0,1]
    return prob