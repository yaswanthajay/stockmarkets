def add_features(df):
    """
    Add technical features to the stock dataframe.

    Args:
        df (pd.DataFrame): Stock price data.

    Returns:
        pd.DataFrame: DataFrame with added moving averages.
    """
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df.dropna(inplace=True)
    return df
