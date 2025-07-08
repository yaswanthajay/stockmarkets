import yfinance as yf
import pandas as pd

def load_and_process_data(ticker="AAPL", period="1y"):
    df = yf.download(ticker, period=period)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)
    return df
