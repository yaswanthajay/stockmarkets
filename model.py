from tensorflow.keras.models import load_model
import numpy as np
import joblib

def load_trained_model():
    return load_model("my_model.h5")

def load_scaler():
    return joblib.load("scaler.save")

def predict(data_dict, model):
    X = data_dict['scaled_input'][-1].reshape(1, 60, 1)
    prediction = model.predict(X, verbose=0)
    return data_dict['scaler'].inverse_transform(prediction)[0][0]
