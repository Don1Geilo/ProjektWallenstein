import os
import sys
from pathlib import Path

import pandas as pd
import pytest

# provide minimal config for main.validate_config
os.environ.setdefault("REDDIT_CLIENT_ID", "test")
os.environ.setdefault("REDDIT_CLIENT_SECRET", "test")
os.environ.setdefault("REDDIT_USER_AGENT", "test")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib

import wallenstein.sentiment as sentiment
import wallenstein.sentiment_analysis as sentiment_analysis


@pytest.fixture(autouse=True)
def _disable_bert(monkeypatch):
    """Ensure keyword-based sentiment to keep tests light."""
    monkeypatch.setattr(sentiment.settings, "USE_BERT_SENTIMENT", False)
    monkeypatch.setattr(
        sentiment_analysis,
        "analyze_sentiment_many",
        lambda texts, batch_size=32: [0.0 for _ in texts],
    )


def test_aggregate_handles_nan_values():
    main = importlib.import_module("main")

    posts = pd.DataFrame(
        [
            {
                "created_utc": 1_700_000_000,
                "ticker": "ABC",
                "title": "bullish",
                "selftext": "",
                "ups": float("nan"),
                "num_comments": float("nan"),
            },
            {
                "created_utc": 1_700_000_000,
                "ticker": "ABC",
                "title": "bearish",
                "selftext": "",
                "ups": 5,
                "num_comments": float("nan"),
            },
        ]
    )

    agg = main.aggregate_daily_sentiment(posts)

    assert agg.loc[0, "n_posts"] == 2
    assert not agg["sentiment_weighted"].isna().any()
