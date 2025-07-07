import yfinance as yf
import pandas as pd
import ta  # Technical Analysis library

def load_and_process_data(ticker):
    df = yf.download(ticker, period="6mo", interval="1d")
    
    # Add technical indicators
    df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['macd'] = ta.trend.MACD(df['Close']).macd()
    df['bollinger_h'] = ta.volatility.BollingerBands(df['Close']).bollinger_hband()
    df['bollinger_l'] = ta.volatility.BollingerBands(df['Close']).bollinger_lband()
    df['volume_avg'] = df['Volume'].rolling(window=10).mean()

    # Drop rows with missing values
    df = df.dropna()
    
    return df
