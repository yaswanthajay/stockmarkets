from tensorflow.keras.models import load_model
import joblib
import numpy as np

def load_trained_model():
    model = load_model("my_model.h5")
    scaler = joblib.load("scaler.save")
    return model, scaler

def predict(data_dict, model, scaler):
    X = data_dict['scaled_input'][-1].reshape(1, 60, 1)
    prediction = model.predict(X)
    return scaler.inverse_transform(prediction)[0][0]
