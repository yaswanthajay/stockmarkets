import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

def train_model(df):
    """
    Train a simple linear regression model to predict closing price
    based on moving averages and optional sentiment.

    Args:
        df (pd.DataFrame): DataFrame with features and 'Close' price.

    Returns:
        model: Trained sklearn model
    """
    features = ['MA10', 'MA20']
    if 'Sentiment' in df.columns:
        features.append('Sentiment')

    X = df[features]
    y = df['Close']

    model = LinearRegression()
    model.fit(X, y)

    joblib.dump(model, 'stock_model.pkl')
    return model
