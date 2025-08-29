import duckdb

from wallenstein.watchlist import (
    add_symbols,
    remove_symbols,
    list_symbols,
    all_unique_symbols,
)


def test_add_list_remove_watchlist():
    con = duckdb.connect(":memory:")

    add_symbols(con, "chat1", ["aapl", "msft"], note="tech")
    assert list_symbols(con, "chat1") == [("AAPL", "tech"), ("MSFT", "tech")]

    add_symbols(con, "chat2", ["msft", "goog"], note=None)
    assert set(all_unique_symbols(con)) == {"AAPL", "GOOG", "MSFT"}

    removed = remove_symbols(con, "chat1", ["aapl"])
    assert removed == 1
    assert list_symbols(con, "chat1") == [("MSFT", "tech")]
    assert set(all_unique_symbols(con)) == {"GOOG", "MSFT"}

    # updating note should replace existing entry
    add_symbols(con, "chat1", ["msft"], note="updated")
    assert list_symbols(con, "chat1") == [("MSFT", "updated")]

    # removing a non-existent symbol should affect nothing
    removed = remove_symbols(con, "chat1", ["tsla"])
    assert removed == 0
