import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_process_data(ticker):
    data = yf.download(ticker, period="6mo", interval="1d")
    data.dropna(inplace=True)
    data = data[['Close']]

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X = np.array([scaled[i-60:i] for i in range(60, len(scaled))])
    return {'raw': data, 'scaled_input': X, 'scaler': scaler}
