# models.py

import os
import joblib

MODELS_DIR = "data/models"

def save_model(model, model_name: str):
    """
    Save the trained model to the models directory.
    :param model: The trained model object
    :param model_name: The name of the model file
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def load_model(model_name: str):
    """
    Load a model from the models directory.
    :param model_name: The name of the model file to load
    :return: The loaded model
    """
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        return model
    else:
        print(f"Model {model_name} not found.")
        return None
