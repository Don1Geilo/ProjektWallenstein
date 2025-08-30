import math
from datetime import datetime, timezone

import duckdb
import pandas as pd
import pytest

# Reuse path setup and environment stubs from existing tests
from test_reddit_scraper import ROOT  # noqa: F401

from wallenstein import reddit_enrich
from wallenstein.db_schema import ensure_tables


def _setup_db():
    con = duckdb.connect(":memory:")
    ensure_tables(con)
    return con


def test_enrich_reddit_posts(monkeypatch):
    con = _setup_db()
    now = datetime.now(timezone.utc)
    posts = {
        "ABC": [
            {
                "id": "abc",
                "created_utc": now,
                "text": "Great stock",
                "upvotes": 9,
            }
        ]
    }
    monkeypatch.setattr(reddit_enrich, "analyze_sentiment", lambda text: 0.5)

    reddit_enrich.enrich_reddit_posts(con, posts, ["ABC"])
    # second call should not create duplicates
    reddit_enrich.enrich_reddit_posts(con, posts, ["ABC"])

    row = con.execute(
        "SELECT id, ticker, sentiment_dict, sentiment_weighted FROM reddit_enriched"
    ).fetchone()

    assert row[0] == int("abc", 36)
    assert row[1] == "ABC"
    assert row[2] == 0.5
    assert row[3] == pytest.approx(0.5 * math.log(10))
    count = con.execute("SELECT COUNT(*) FROM reddit_enriched").fetchone()[0]
    assert count == 1


def test_compute_reddit_trends():
    con = _setup_db()
    df = pd.DataFrame(
        [
            {
                "id": 1,
                "ticker": "ABC",
                "created_utc": datetime(2024, 1, 1, 12, tzinfo=timezone.utc),
                "text": "",
                "upvotes": 1,
                "sentiment_dict": 0.1,
                "sentiment_weighted": 0.1,
                "sentiment_ml": None,
                "return_1d": None,
                "return_3d": None,
                "return_7d": None,
            },
            {
                "id": 2,
                "ticker": "ABC",
                "created_utc": datetime(2024, 1, 1, 18, tzinfo=timezone.utc),
                "text": "",
                "upvotes": 3,
                "sentiment_dict": 0.2,
                "sentiment_weighted": 0.2,
                "sentiment_ml": None,
                "return_1d": None,
                "return_3d": None,
                "return_7d": None,
            },
        ]
    )
    con.register("df", df)
    con.execute("INSERT INTO reddit_enriched SELECT * FROM df")

    reddit_enrich.compute_reddit_trends(con)

    row = con.execute(
        "SELECT date, ticker, mentions, avg_upvotes, hotness FROM reddit_trends"
    ).fetchone()

    assert str(row[0]) == "2024-01-01"
    assert row[1] == "ABC"
    assert row[2] == 2
    assert row[3] == pytest.approx(2.0)
    assert row[4] == pytest.approx(4.0)


def test_compute_returns():
    con = _setup_db()
    post_df = pd.DataFrame(
        [
            {
                "id": 1,
                "ticker": "ABC",
                "created_utc": datetime(2024, 1, 1, tzinfo=timezone.utc),
                "text": "",
                "upvotes": 0,
                "sentiment_dict": 0.1,
                "sentiment_weighted": 0.1,
                "sentiment_ml": None,
                "return_1d": None,
                "return_3d": None,
                "return_7d": None,
            }
        ]
    )
    prices_df = pd.DataFrame(
        [
            {
                "date": datetime(2024, 1, 1),
                "ticker": "ABC",
                "open": 0,
                "high": 0,
                "low": 0,
                "close": 10,
                "adj_close": 0,
                "volume": 0,
            },
            {
                "date": datetime(2024, 1, 2),
                "ticker": "ABC",
                "open": 0,
                "high": 0,
                "low": 0,
                "close": 11,
                "adj_close": 0,
                "volume": 0,
            },
            {
                "date": datetime(2024, 1, 4),
                "ticker": "ABC",
                "open": 0,
                "high": 0,
                "low": 0,
                "close": 14,
                "adj_close": 0,
                "volume": 0,
            },
            {
                "date": datetime(2024, 1, 8),
                "ticker": "ABC",
                "open": 0,
                "high": 0,
                "low": 0,
                "close": 20,
                "adj_close": 0,
                "volume": 0,
            },
        ]
    )
    con.register("posts", post_df)
    con.execute("INSERT INTO reddit_enriched SELECT * FROM posts")
    con.register("prices_df", prices_df)
    con.execute("INSERT INTO prices SELECT * FROM prices_df")

    reddit_enrich.compute_returns(con)

    row = con.execute(
        "SELECT return_1d, return_3d, return_7d FROM reddit_enriched WHERE id=1"
    ).fetchone()

    assert row[0] == pytest.approx(0.1)
    assert row[1] == pytest.approx(0.4)
    assert row[2] == pytest.approx(1.0)
