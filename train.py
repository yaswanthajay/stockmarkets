from fetch_data import load_and_process_data
from prepare_data import create_lstm_dataset
from model import train_and_save_model

ticker = "AAPL"
data = load_and_process_data(ticker)
features = ['Close', 'rsi', 'macd', 'bollinger_h', 'bollinger_l', 'volume_avg']

X, y = create_lstm_dataset(data, feature_cols=features, target_col='Close')
train_and_save_model(X, y, model_path='lstm_stock_model.h5')
