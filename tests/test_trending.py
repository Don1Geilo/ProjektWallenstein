from datetime import datetime, timedelta, timezone

import duckdb
import pytest

from wallenstein.aliases import add_alias
from wallenstein.db_schema import ensure_tables
from wallenstein.reddit_scraper import detect_trending_tickers
import wallenstein.trending as trending
from wallenstein.ticker_detection import TickerMetadata
from wallenstein.trending import scan_reddit_for_candidates


@pytest.fixture(autouse=True)
def disable_discovery(monkeypatch):
    monkeypatch.setattr(trending, "discover_new_tickers", lambda *args, **kwargs: {})


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


def test_scan_candidates_promotes_discovered_symbol(monkeypatch):
    con = duckdb.connect(database=":memory:")
    ensure_tables(con)

    now = datetime.now(timezone.utc)
    rows = [
        ("p1", now - timedelta(hours=1), "🚀 $NEW to the moon", "", 42),
        ("p2", now - timedelta(hours=2), "$NEW again", "", 30),
    ]
    _insert_posts(con, rows)

    meta = TickerMetadata(symbol="NEW", aliases={"New Holdings"})

    def fake_discover(texts, known=None, **kwargs):
        return {"NEW": meta}

    monkeypatch.setattr(trending, "discover_new_tickers", fake_discover)

    candidates = scan_reddit_for_candidates(
        con,
        lookback_days=2,
        window_hours=24,
        min_mentions=1,
        min_lift=1.0,
    )

    new_candidate = next(c for c in candidates if c.symbol == "NEW")
    assert new_candidate.is_known is True

    aliases = con.execute(
        "SELECT alias FROM ticker_aliases WHERE ticker = 'NEW' ORDER BY alias"
    ).fetchall()
    stored = {row[0] for row in aliases}
    assert {"NEW", "New Holdings"} <= stored


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


def test_scan_candidates_adds_weekly_return_from_prices():
    con = duckdb.connect(database=":memory:")
    ensure_tables(con)

    add_alias(con, "TSLA", "tesla")
    now = datetime.now(timezone.utc)
    rows = [
        ("p1", now - timedelta(hours=1), "$TSLA rockets", "", 18),
        ("p2", now - timedelta(hours=3), "Holding TSLA", "", 7),
    ]
    _insert_posts(con, rows)

    price_rows = [
        (
            (now - timedelta(days=7)).date(),
            "TSLA",
            100.0,
            105.0,
            99.0,
            102.0,
            102.0,
            1_000,
        ),
        (
            now.date(),
            "TSLA",
            110.0,
            115.0,
            108.0,
            112.0,
            112.0,
            1_200,
        ),
    ]
    con.executemany(
        """
        INSERT INTO prices (date, ticker, open, high, low, close, adj_close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        price_rows,
    )

    candidates = scan_reddit_for_candidates(
        con,
        lookback_days=2,
        window_hours=24,
        min_mentions=1,
        min_lift=1.0,
    )

    tsla_candidate = next(c for c in candidates if c.symbol == "TSLA")
    assert tsla_candidate.weekly_return is not None
    expected = 112.0 / 102.0 - 1
    assert tsla_candidate.weekly_return == pytest.approx(expected, rel=1e-3)


def test_scan_candidates_prefers_display_order_for_weekly(monkeypatch):
    """Top candidates should receive weekly returns even with many symbols."""

    con = duckdb.connect(database=":memory:")
    ensure_tables(con)

    tickers = [
        "AAPL",
        "AMD",
        "AMZN",
        "BABA",
        "GME",
        "GOOG",
        "META",
        "MSFT",
        "NIO",
        "NVDA",
        "PLTR",
        "TSLA",

    ]

    now = datetime.now(timezone.utc)
    for t in tickers:
        add_alias(con, t, t.lower())

    rows = []
    for idx, ticker in enumerate(tickers):
        for j in range(idx + 1):
            rows.append(
                (
                    f"{ticker}{j}",
                    now - timedelta(hours=j + 1),
                    f"$ {ticker}",
                    "",
                    5,
                )
            )
    _insert_posts(con, rows)

    monkeypatch.setattr(trending, "_weekly_return_from_db", lambda *_: None)
    weekly_values = {"TSLA": 0.42, "PLTR": 0.21}
    monkeypatch.setattr(
        trending,
        "_weekly_return_from_yfinance",
        lambda symbol: weekly_values.get(symbol, 0.0),
    )

    candidates = scan_reddit_for_candidates(
        con,
        lookback_days=7,
        window_hours=24,
        min_mentions=1,
        min_lift=1.0,
    )

    top_symbols = [c.symbol for c in candidates[:2]]
    assert "TSLA" in top_symbols and "PLTR" in top_symbols

    for sym in ("TSLA", "PLTR"):
        cand = next(c for c in candidates if c.symbol == sym)
        assert cand.weekly_return == pytest.approx(weekly_values[sym])

