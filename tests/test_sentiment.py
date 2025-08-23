import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import wallenstein.sentiment as sentiment
from wallenstein.sentiment import (
    analyze_sentiment,
    analyze_sentiment_bert,
    aggregate_sentiment_by_ticker,
    derive_recommendation,
)


@pytest.fixture(autouse=True)
def _disable_bert(monkeypatch):
    """Use keyword sentiment by default for tests."""

    monkeypatch.setenv("USE_BERT_SENTIMENT", "0")

def test_analyze_sentiment_keywords():
    text = "I'm going long and want to buy more calls, not sell"
    assert analyze_sentiment(text) > 0
    text2 = "Time to sell and go short, very bearish"
    assert analyze_sentiment(text2) < 0
    text3 = "Lasst uns jetzt kaufen, das ist sehr bullisch"
    assert analyze_sentiment(text3) > 0
    text4 = "Vielleicht sollten wir verkaufen, es wirkt bärisch"
    assert analyze_sentiment(text4) < 0


def test_aggregate_and_recommendation():
    data = {
        "NVDA": [{"text": "buy the dip"}, {"text": "bullish"}, {"text": "kaufen"}],
        "AMZN": [{"text": "sell now"}, {"text": "verkaufen"}]
    }
    scores = aggregate_sentiment_by_ticker(data)
    assert scores["NVDA"] > 0
    assert scores["AMZN"] < 0
    assert derive_recommendation(scores["NVDA"]) == "Buy"
    assert derive_recommendation(scores["AMZN"]) == "Sell"
    assert derive_recommendation(0.0) == "Hold"


def test_intensity_and_negation():
    assert analyze_sentiment("This is a strong buy signal") == 2
    assert analyze_sentiment("Bitte nicht kaufen") < 0



def test_analyze_sentiment_bert_mock():
    with patch("wallenstein.sentiment.BertSentiment") as MockBert:
        sentiment._bert_analyzer = None
        instance = MockBert.return_value
        instance.return_value = [{"label": "positive", "score": 0.8}]
        assert analyze_sentiment_bert("text") > 0
        instance.return_value = [{"label": "negative", "score": 0.6}]
        assert analyze_sentiment_bert("text") < 0


def test_env_switches_to_bert(monkeypatch):
    with patch("wallenstein.sentiment.BertSentiment") as MockBert:
        sentiment._bert_analyzer = None
        MockBert.return_value.return_value = [{"label": "positive", "score": 0.9}]
        monkeypatch.setenv("USE_BERT_SENTIMENT", "1")
        assert analyze_sentiment("whatever") > 0
        monkeypatch.delenv("USE_BERT_SENTIMENT", raising=False)
        # ensure fallback path still works
        assert analyze_sentiment("buy") > 0

def test_negation_with_filler_tokens():
    assert analyze_sentiment("nicht so bullish") < 0
    assert analyze_sentiment("kein kauf heute") < 0

