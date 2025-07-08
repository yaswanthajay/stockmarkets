import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import joblib

# Step 1: Generate dummy time series data (you can replace this with real stock data)
data = np.linspace(100, 200, 500).reshape(-1, 1)

# Step 2: Scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Step 3: Create sequences (60 time steps to 1 output)
X, y = [], []
for i in range(60, len(scaled_data)):
    X.append(scaled_data[i-60:i])
    y.append(scaled_data[i])
X, y = np.array(X), np.array(y)

# Step 4: Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(60, 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Step 5: Train model
model.fit(X, y, epochs=10, batch_size=32)

# Step 6: Save model and scaler
model.save("my_model.h5")
joblib.dump(scaler, "scaler.save")

print("✅ Model and scaler saved as 'my_model.h5' and 'scaler.save'")
