import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Load data
data = yf.download("AAPL", start="2015-01-01", end="2023-12-31")
closing_prices = data["Close"].values.reshape(-1, 1)

# Scale data
scaler = MinMaxScaler()
scaled_prices = scaler.fit_transform(closing_prices)

# Prepare sequences
X, y = [], []
sequence_length = 60
for i in range(sequence_length, len(scaled_prices)):
    X.append(scaled_prices[i - sequence_length:i])
    y.append(scaled_prices[i])

X, y = np.array(X), np.array(y)

# Build model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")

# Train model
early_stop = EarlyStopping(monitor='loss', patience=5)
model.fit(X, y, epochs=10, batch_size=32, callbacks=[early_stop])

# Save model
model.save("my_model.h5")
