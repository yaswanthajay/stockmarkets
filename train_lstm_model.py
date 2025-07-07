import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import ta  # Technical Analysis library
import joblib

def load_data(ticker="AAPL"):
    df = yf.download(ticker, period="6mo", interval="1d")
    df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['macd'] = ta.trend.MACD(df['Close']).macd()
    bb = ta.volatility.BollingerBands(df['Close'])
    df['bollinger_h'] = bb.bollinger_hband()
    df['bollinger_l'] = bb.bollinger_lband()
    df['volume_avg'] = df['Volume'].rolling(window=10).mean()
    df.dropna(inplace=True)
    return df

def create_dataset(df, features, target='Close', look_back=20):
    X, y = [], []
    for i in range(look_back, len(df)):
        X.append(df[features].iloc[i-look_back:i].values)
        y.append(df[target].iloc[i])
    return np.array(X), np.array(y)

# Load and prepare data
features = ['Close', 'rsi', 'macd', 'bollinger_h', 'bollinger_l', 'volume_avg']
df = load_data("AAPL")
X, y = create_dataset(df, features)

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
early_stop = EarlyStopping(monitor='loss', patience=5)
model.fit(X, y, epochs=20, batch_size=32, callbacks=[early_stop])

# Save model
model.save("stockmodel.h5")
