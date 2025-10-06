import duckdb

from wallenstein.aliases import add_alias
from wallenstein.trending import ensure_trending_tables, auto_add_candidates_to_watchlist


def test_auto_added_trends_use_global_watchlist_chat_id():
    con = duckdb.connect(":memory:")

    # ensure dependent tables exist
    ensure_trending_tables(con)
    add_alias(con, "HOT", "Hot Corp")

    con.execute(
        """
        INSERT INTO trending_candidates (
            symbol, mentions_24h, baseline_rate_per_h, lift, trend, first_seen, last_seen
        ) VALUES (?, ?, ?, ?, ?, NOW(), NOW())
        """,
        ("HOT", 40, 1.0, 5.0, 9.5),
    )

    added = auto_add_candidates_to_watchlist(
        con,
        notify_fn=None,
        max_new=3,
        min_mentions=10,
        min_lift=2.0,
    )

    assert added == ["HOT"]
    rows = con.execute(
        "SELECT chat_id, symbol FROM watchlist WHERE symbol = 'HOT'"
    ).fetchall()
    assert rows == [("_GLOBAL_", "HOT")]


def test_auto_add_initialises_watchlist_table_if_missing():
    con = duckdb.connect(":memory:")

    ensure_trending_tables(con)
    add_alias(con, "AMD", "AMD")
    con.execute(
        """
        INSERT INTO trending_candidates (
            symbol, mentions_24h, baseline_rate_per_h, lift, trend, first_seen, last_seen
        ) VALUES (?, ?, ?, ?, ?, NOW(), NOW())
        """,
        ("AMD", 35, 1.0, 5.0, 8.7),
    )

    added = auto_add_candidates_to_watchlist(
        con,
        notify_fn=None,
        max_new=5,
        min_mentions=10,
        min_lift=2.0,
    )

    assert added == ["AMD"]
    assert con.execute("SELECT chat_id FROM watchlist WHERE symbol = 'AMD'").fetchall() == [
        ("_GLOBAL_",)
    ]
