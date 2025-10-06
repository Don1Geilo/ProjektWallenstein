import duckdb

from wallenstein.aliases import add_alias
from wallenstein.trending import ensure_trending_tables, auto_add_candidates_to_watchlist
from wallenstein.watchlist import add_symbols


def test_auto_added_trends_use_global_watchlist_chat_id():
    con = duckdb.connect(":memory:")

    # ensure dependent tables exist
    add_symbols(con, "_init", [])
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
