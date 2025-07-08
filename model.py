from tensorflow.keras.models import load_model
import numpy as np

def load_trained_model(path='my_model.h5'):
    return load_model(path)

def predict(data_dict, model):
    X = data_dict['scaled_input'][-1].reshape(1, 60, 1)
    prediction = model.predict(X)
    return data_dict['scaler'].inverse_transform(prediction)[0][0]
