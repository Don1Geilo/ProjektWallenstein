from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Einmalige Installation des NLTK-Moduls
# nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    """Gibt das Sentiment eines Textes zurück (-1 bis +1)."""
    sentiment = sia.polarity_scores(text)
    return sentiment["compound"]
