import yfinance as yf
import pandas as pd
import ta

def load_and_process_data(ticker, period="1y"):
    df = yf.download(ticker, period=period, progress=False)
    df.dropna(inplace=True)

    df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['macd'] = ta.trend.MACD(df['Close']).macd_diff()
    bb = ta.volatility.BollingerBands(df['Close'])
    df['bollinger_h'] = bb.bollinger_hband()
    df['bollinger_l'] = bb.bollinger_lband()
    df['volume_avg'] = df['Volume'].rolling(20).mean()

    df.dropna(inplace=True)
    return df
