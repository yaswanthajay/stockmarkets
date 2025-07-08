# app.py
import streamlit as st
import pandas as pd
import numpy as np
from fetch_data import load_and_process_data
from model import load_trained_model
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="📈 Stock Predictor - MarketPulse Style", layout="centered")

st.title("📊 Stock Market Predictor App")
st.markdown("Enter a stock ticker (like `AAPL`, `TSLA`, `GOOG`) to get a prediction for the next day's price.")

ticker = st.text_input("Stock Ticker:", "AAPL")

if st.button("Predict"):
    try:
        df = load_and_process_data(ticker)
        model = load_trained_model()
        data = df[['Close']].values

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        last_60_days = scaled_data[-60:]
        X_test = np.reshape(last_60_days, (1, 60, 1))

        prediction = model.predict(X_test)
        predicted_price = scaler.inverse_transform(prediction)

        st.success(f"📈 Predicted Next Price: **${predicted_price[0][0]:.2f}**")
        st.line_chart(df['Close'])

    except FileNotFoundError:
        st.error("❌ Model file not found! Please run `train_model.py` to generate `my_model.h5`.")
    except Exception as e:
        st.error(f"🚨 Error occurred: {str(e)}")
