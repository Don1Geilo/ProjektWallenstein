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
    analyze_sentiment_batch,
    aggregate_sentiment_by_ticker,
    derive_recommendation,
)


@pytest.fixture(autouse=True)
def _disable_bert(monkeypatch):
    """Use keyword sentiment by default for tests."""
    monkeypatch.setattr(sentiment.settings, "USE_BERT_SENTIMENT", False)

def test_analyze_sentiment_keywords():
    text = "I'm going long and want to buy more calls, not sell"
    assert analyze_sentiment(text) > 0
    text2 = "Time to sell and go short, very bearish"
    assert analyze_sentiment(text2) < 0
    text3 = "Lasst uns jetzt kaufen, das ist sehr bullisch"
    assert analyze_sentiment(text3) > 0
    text4 = "Vielleicht sollten wir verkaufen, es wirkt bärisch"
    assert analyze_sentiment(text4) < 0
    text5 = "Der Gewinn ist heute grün"
    assert analyze_sentiment(text5) > 0
    text6 = "Der Verlust ist rot"
    assert analyze_sentiment(text6) < 0


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
    assert analyze_sentiment("Das ist ein mega kauf") == 2
    assert analyze_sentiment("Bitte nicht kaufen") < 0
    assert analyze_sentiment("nie kaufen") < 0



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
        monkeypatch.setattr(sentiment.settings, "USE_BERT_SENTIMENT", True)
        assert analyze_sentiment("whatever") > 0
        monkeypatch.setattr(sentiment.settings, "USE_BERT_SENTIMENT", False)
        # ensure fallback path still works
        assert analyze_sentiment("buy") > 0

def test_negation_with_filler_tokens():
    assert analyze_sentiment("nicht so bullish") < 0
    assert analyze_sentiment("kein kauf heute") < 0


def test_analyze_sentiment_batch_keyword():
    texts = ["bullish", "verkaufen"]
    scores = analyze_sentiment_batch(texts)
    assert scores[0] > 0
    assert scores[1] < 0


def test_keywords_loaded_from_file():
    import json, importlib

    kw_file = ROOT / "data" / "sentiment_keywords.json"
    original = kw_file.read_text() if kw_file.exists() else None
    try:
        kw_file.write_text(json.dumps({"superbull": 1, "schrott": -1}))
        importlib.reload(sentiment)
        assert sentiment.KEYWORD_SCORES["superbull"] == 1
        assert sentiment.KEYWORD_SCORES["schrott"] == -1
    finally:
        if original is None:
            kw_file.unlink()
        else:
            kw_file.write_text(original)
        importlib.reload(sentiment)


def test_markers_loaded_from_config():
    import importlib
    pytest.importorskip("yaml")

    cfg_file = ROOT / "data" / "sentiment_config.yaml"
    original = cfg_file.read_text() if cfg_file.exists() else None
    try:
        cfg_file.write_text(
            "intensity_weights:\n  hyper: 3\nnegation_markers:\n  - jamais\n"
        )
        importlib.reload(sentiment)
        assert sentiment.INTENSITY_WEIGHTS["hyper"] == 3
        assert "jamais" in sentiment.NEGATION_MARKERS
    finally:
        if original is None:
            cfg_file.unlink()
        else:
            cfg_file.write_text(original)
        importlib.reload(sentiment)
