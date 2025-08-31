import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wallenstein.sentiment_analysis import SentimentEngine


def test_pipeline_uses_top_k_none(monkeypatch):
    captured = {}

    class DummyPipeline:
        def __init__(self, *args, **kwargs):
            captured["top_k"] = kwargs.get("top_k")

        def __call__(self, text):
            return [[{"label": "positive", "score": 0.8}, {"label": "negative", "score": 0.2}]]

    monkeypatch.setattr("wallenstein.sentiment_analysis._try_import_transformers", lambda: DummyPipeline)
    monkeypatch.setattr("wallenstein.sentiment_analysis._ensure_vader", lambda: None)
    monkeypatch.setattr("wallenstein.sentiment_analysis._detect_lang", lambda text: "en")
    engine = SentimentEngine()
    res = engine.analyze("good")
    assert captured["top_k"] is None
    assert res.meta["scores"]["positive"] == pytest.approx(0.8, abs=0.01)
    assert res.meta["scores"]["negative"] == pytest.approx(0.2, abs=0.01)
