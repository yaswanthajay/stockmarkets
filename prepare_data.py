import numpy as np

def create_lstm_dataset(data, feature_cols, target_col='Close', seq_length=30):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[feature_cols].iloc[i:i+seq_length].values)
        y.append(data[target_col].iloc[i+seq_length])
    return np.array(X), np.array(y)
