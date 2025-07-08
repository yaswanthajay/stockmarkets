import streamlit as st
import numpy as np
from model import load_trained_model, predict, load_scaler

st.title("📈 Stock Price Predictor")

model = load_trained_model()
scaler = load_scaler()

# Generate dummy input
data = np.linspace(150, 180, 60).reshape(-1, 1)
scaled_input = scaler.transform(data)

data_dict = {
    'scaled_input': scaled_input.reshape(1, 60, 1),
    'scaler': scaler
}

if st.button("Predict"):
    pred = predict(data_dict, model)
    st.success(f"Predicted Stock Price: ₹{pred:.2f}")
