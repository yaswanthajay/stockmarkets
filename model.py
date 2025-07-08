from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_trained_model(model_path: str = "my_model.h5"):
    """Loads a trained Keras model."""
    try:
        model = load_model(model_path)
        print(f"✅ Model loaded from {model_path}")
        return model
    except Exception as e:
        raise FileNotFoundError(f"❌ Error loading model: {e}")

def predict(data_dict: dict, model):
    """
    Predict the next value using the trained model and last 60 timesteps.

    Args:
        data_dict: Dictionary with keys:
            - 'scaled_input': np.array of shape (N, 60, 1)
            - 'scaler': fitted MinMaxScaler
        model: Trained Keras model

    Returns:
        float: predicted stock price (after inverse transform)
    """
    try:
        if 'scaled_input' not in data_dict or 'scaler' not in data_dict:
            raise ValueError("Missing 'scaled_input' or 'scaler' in data_dict.")

        X = np.array(data_dict['scaled_input'])[-1].reshape(1, 60, 1)
        prediction = model.predict(X, verbose=0)
        predicted_price = data_dict['scaler'].inverse_transform(prediction)[0][0]
        return float(predicted_price)

    except Exception as e:
        raise RuntimeError(f"❌ Prediction failed: {e}")
