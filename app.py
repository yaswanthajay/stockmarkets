import streamlit as st
import pandas as pd
import joblib

from fetch_data import load_stock_data
from features import add_features
from news_sentiment import analyze_sentiment
from model import train_model

st.set_page_config(page_title="📈 MarketPulse AI", layout="centered")
st.title("📈 AI Stock Market Predictor with Explainability")
st.markdown("Enter a stock ticker to view predictions based on past data + sentiment.")

stock = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA):", value="AAPL")

if st.button("🔮 Train Model & Predict"):
    with st.spinner("Training model and predicting..."):
        try:
            # Load & process data
            data = load_stock_data(stock)
            data = add_features(data)

            # Add sentiment score (dummy here)
            sentiment_score = analyze_sentiment(stock)
            data['Sentiment'] = sentiment_score

            # Train and save model
            model = train_model(data)

            # Predict using the trained model
            features = ['MA10', 'MA20', 'Sentiment']
            data['Prediction'] = model.predict(data[features])

            st.success("✅ Model trained and prediction done!")
            st.line_chart(data[['Close', 'MA10', 'MA20', 'Prediction']])

        except Exception as e:
            st.error(f"❌ Error: {e}")

st.caption("🕒 Model trains & updates daily for best accuracy.")
