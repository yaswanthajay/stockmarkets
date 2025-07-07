import numpy as np

def create_lstm_dataset(df, feature_cols, target_col, look_back=20):
    X, y = [], []
    for i in range(look_back, len(df)):
        X.append(df[feature_cols].iloc[i-look_back:i].values)
        y.append(df[target_col].iloc[i])
    return np.array(X), np.array(y)
