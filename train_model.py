import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam

# Dummy time series data for training (replace with real stock data for production)
X = np.random.rand(1000, 60, 1)
y = np.random.rand(1000, 1)

model = Sequential([
    LSTM(64, input_shape=(60, 1)),
    Dense(1)
])

model.compile(optimizer=Adam(0.001), loss='mse')
model.fit(X, y, epochs=5, batch_size=32)

model.save("my_model.h5")
