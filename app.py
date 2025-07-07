import streamlit as st
import numpy as np
from fetch_data import load_and_process_data
from prepare_data import create_lstm_dataset
from model import load_trained_model

st.set_page_config(page_title="📈 MarketPulse AI", layout="centered")
st.title("📈 AI-Powered Stock Market Predictor")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA)", value="AAPL")

if st.button("🔍 Predict Next Closing Price"):
    try:
        data = load_and_process_data(ticker)
        features = ['Close', 'rsi', 'macd', 'bollinger_h', 'bollinger_l', 'volume_avg']
        X, y = create_lstm_dataset(data, feature_cols=features, target_col='Close')

        model = load_trained_model()
        next_pred = model.predict(X[-1].reshape(1, X.shape[1], X.shape[2]))
        st.success(f"📊 Predicted next closing price: **${next_pred[0][0]:.2f}**")

        st.line_chart(data[['Close']])

    except Exception as e:
        st.error(f"❌ Error: {e}")
