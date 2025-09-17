from datetime import datetime, timedelta, timezone

import duckdb

from wallenstein.aliases import add_alias
from wallenstein.db_schema import ensure_tables
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

    trending = detect_trending_tickers(
        data, window_hours=24, baseline_days=7, min_mentions=3, ratio=2.0
    )

    assert "AAA" in trending
    assert "BBB" not in trending


def _insert_posts(con: duckdb.DuckDBPyConnection, rows: list[tuple]):
    con.executemany(
        """
        INSERT INTO reddit_posts (id, created_utc, title, text, upvotes)
        VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )


def test_scan_candidates_detects_new_symbol_marked_unknown():
    con = duckdb.connect(database=":memory:")
    ensure_tables(con)

    now = datetime.now(timezone.utc)
    rows = [
        ("p1", now - timedelta(hours=1), "🚀 $NEW to the moon", "", 42),
        ("p2", now - timedelta(hours=2), "$NEW again", "", 30),
    ]
    _insert_posts(con, rows)

    candidates = scan_reddit_for_candidates(
        con,
        lookback_days=2,
        window_hours=24,
        min_mentions=1,
        min_lift=1.0,
    )

    new_candidate = next(c for c in candidates if c.symbol == "NEW")
    assert new_candidate.is_known is False


def test_scan_candidates_marks_known_symbols():
    con = duckdb.connect(database=":memory:")
    ensure_tables(con)

    add_alias(con, "TSLA", "tesla")
    now = datetime.now(timezone.utc)
    rows = [
        ("p1", now - timedelta(hours=1), "$TSLA to the moon", "", 15),
        ("p2", now - timedelta(hours=3), "Holding TSLA", "", 10),
    ]
    _insert_posts(con, rows)

    candidates = scan_reddit_for_candidates(
        con,
        lookback_days=2,
        window_hours=24,
        min_mentions=1,
        min_lift=1.0,
    )

    tsla_candidate = next(c for c in candidates if c.symbol == "TSLA")
    assert tsla_candidate.is_known is True


def test_scan_candidates_handles_dot_symbol():
    con = duckdb.connect(database=":memory:")
    ensure_tables(con)

    add_alias(con, "BRK.B", "berkshire")
    now = datetime.now(timezone.utc)
    rows = [
        ("p1", now - timedelta(hours=1), "Buying $BRK.B today", "", 12),
        ("p2", now - timedelta(hours=2), "Long BRK.B", "", 8),
    ]
    _insert_posts(con, rows)

    candidates = scan_reddit_for_candidates(
        con,
        lookback_days=2,
        window_hours=24,
        min_mentions=1,
        min_lift=1.0,
    )

    brkb_candidate = next(c for c in candidates if c.symbol == "BRK.B")
    assert brkb_candidate.is_known is True
