from datetime import datetime, timedelta, timezone

import duckdb

from wallenstein.reddit_scraper import detect_trending_tickers
from wallenstein.trending import scan_reddit_for_candidates


def test_detect_trending_tickers():
    now = datetime.now(timezone.utc)
    data = {
        "AAA": [
            {"created_utc": now - timedelta(hours=1), "text": "aaa"},
            {"created_utc": now - timedelta(hours=2), "text": "aaa"},
            {"created_utc": now - timedelta(hours=3), "text": "aaa"},
        ],
        "BBB": [
            {"created_utc": now - timedelta(days=2), "text": "bbb"},
            {"created_utc": now - timedelta(days=3), "text": "bbb"},
        ],
    }
    trending = detect_trending_tickers(data, window_hours=24, baseline_days=7, min_mentions=3, ratio=2.0)
    assert "AAA" in trending
    assert "BBB" not in trending


def test_scan_reddit_for_candidates_finds_new_symbol(tmp_path):
    db_path = tmp_path / "db.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(
        "CREATE TABLE reddit_posts (id VARCHAR, created_utc TIMESTAMP, title VARCHAR, text VARCHAR, upvotes INTEGER)"
    )
    now = datetime.utcnow()
    con.execute(
        "INSERT INTO reddit_posts VALUES (?, ?, ?, ?, ?)",
        ("p1", now, "🚀 $NEW to the moon", "", 42),
    )

    candidates = scan_reddit_for_candidates(
        con,
        lookback_days=3,
        window_hours=24,
        min_mentions=1,
        min_lift=1.0,
    )

    assert any(c.symbol == "NEW" for c in candidates)
    con.close()
