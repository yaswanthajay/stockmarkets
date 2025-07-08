from tensorflow.keras.models import load_model
import numpy as np

def load_trained_model():
    return load_model("my_model.h5")

    

def predict(data_dict, model):
    X = data_dict['scaled_input'][-1].reshape(1, 60, 1)
    prediction = model.predict(X)
    return data_dict['scaler'].inverse_transform(prediction)[0][0]
