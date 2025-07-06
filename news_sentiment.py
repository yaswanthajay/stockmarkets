from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def analyze_sentiment(ticker):
    """
    Dummy function to return sentiment score for a stock.

    In real app, you'd fetch news or tweets and analyze sentiment.
    Here, it returns neutral sentiment (0) as placeholder.

    Args:
        ticker (str): Stock symbol.

    Returns:
        float: Sentiment score between -1 (negative) and 1 (positive).
    """
    analyzer = SentimentIntensityAnalyzer()
    # Normally, you fetch news text and analyze:
    # For now return 0 (neutral)
    return 0.0
