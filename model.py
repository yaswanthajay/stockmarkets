from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_and_save_model(X_train, y_train, model_path='lstm_stock_model.h5'):
    model = build_lstm_model(X_train.shape[1:])
    model.fit(X_train, y_train, epochs=20, batch_size=32)
    model.save(model_path)
    return model

def load_trained_model(model_path='lstm_stock_model.h5'):
    return load_model(model_path)
