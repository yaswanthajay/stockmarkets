import yfinance as yf

def load_stock_data(ticker, period='1y', interval='1d'):
    """
    Fetch historical stock data from Yahoo Finance.

    Args:
        ticker (str): Stock symbol, e.g., 'AAPL'.
        period (str): Data period (e.g., '1y' for 1 year).
        interval (str): Data frequency (e.g., '1d' for daily).

    Returns:
        pd.DataFrame: Stock data with columns like Open, High, Low, Close, Volume.
    """
    data = yf.download(ticker, period=period, interval=interval)
    data.dropna(inplace=True)
    return data
