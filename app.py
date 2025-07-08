import streamlit as st
from fetch_data import load_and_process_data
from model import load_trained_model, predict
import pandas as pd
import numpy as np

st.set_page_config(page_title="📈 Stock Predictor", layout="centered")
st.title("📊 Stock Market Predictor")
st.write("Enter a stock symbol (e.g., AAPL, MSFT) to predict next day's price.")

ticker = st.text_input("Stock Symbol", "AAPL")

if st.button("Predict"):
    try:
        df = load_and_process_data(ticker)
        model = load_trained_model()
        price = predict(df, model)
        st.success(f"Predicted price for tomorrow: **${price:.2f}**")
        st.line_chart(df['Close'])
    except Exception as e:
        st.error(f"Error: {e}")
