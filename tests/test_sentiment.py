import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wallenstein.sentiment import analyze_sentiment, aggregate_sentiment_by_ticker, derive_recommendation

def test_analyze_sentiment_keywords():
    text = "I'm going long and want to buy more calls, not sell"
    assert analyze_sentiment(text) > 0
    text2 = "Time to sell and go short, very bearish"
    assert analyze_sentiment(text2) < 0


def test_aggregate_and_recommendation():
    data = {
        "NVDA": [{"text": "buy the dip"}, {"text": "bullish"}],
        "AMZN": [{"text": "sell now"}]
    }
    scores = aggregate_sentiment_by_ticker(data)
    assert scores["NVDA"] > 0
    assert scores["AMZN"] < 0
    assert derive_recommendation(scores["NVDA"]) == "Buy"
    assert derive_recommendation(scores["AMZN"]) == "Sell"
    assert derive_recommendation(0.0) == "Hold"
