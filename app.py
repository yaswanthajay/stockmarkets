import streamlit as st
import pandas as pd
import numpy as np
from fetch_data import load_and_process_data
from model import load_trained_model
from sklearn.preprocessing import MinMaxScaler

st.title("📈 Stock Market Predictor – MarketPulse Clone")

ticker = st.text_input("Enter stock symbol:", "AAPL")

if st.button("Predict"):
    df = load_and_process_data(ticker)
    model = load_trained_model()
    
    # Preprocessing for prediction
    data = df[['Close']].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    last_60_days = scaled_data[-60:]
    X_test = np.reshape(last_60_days, (1, 60, 1))

    prediction = model.predict(X_test)
    predicted_price = scaler.inverse_transform(prediction)

    st.success(f"📊 Predicted price: {predicted_price[0][0]:.2f} USD")

    st.line_chart(df['Close'])
