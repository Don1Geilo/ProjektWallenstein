from datetime import datetime, timedelta, timezone

from wallenstein.reddit_scraper import detect_trending_tickers


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
