import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

def load_stock_data(stock="AAPL"):
    df = yf.download(stock, start="2018-01-01", end="2023-12-31")
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df.dropna(inplace=True)
    return df

def train_model():
    df = load_stock_data()
    X = df[['MA10', 'MA20']]
    y = df['Close']
    
    model = RandomForestRegressor()
    model.fit(X, y)
    
    joblib.dump(model, "stock_model.pkl")
    print("✅ Model saved as stock_model.pkl")

if __name__ == "__main__":
    train_model()
