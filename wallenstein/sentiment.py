def analyze_sentiment(text: str) -> float:
    """
    Dummy-Sentiment: positive bei 'long/call', negativ bei 'short/put'
    """
    text = text.lower()
    score = 0
    if "long" in text or "call" in text:
        score += 1
    if "short" in text or "put" in text:
        score -= 1
    return score
