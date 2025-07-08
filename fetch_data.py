import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator

def load_and_process_data(symbol, days):
    df = yf.download(symbol, period=f"{days}d")
    if df.empty:
        raise ValueError("No data found for the symbol.")
    df['RSI'] = RSIIndicator(df['Close']).rsi()
    df.fillna(method='bfill', inplace=True)
    return df
