import duckdb
import pandas as pd

from wallenstein.correlations import compute_price_sentiment_correlations


def test_compute_price_sentiment_correlations_returns_values(tmp_path):
    db_path = tmp_path / "corr.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute("CREATE TABLE prices (date DATE, ticker VARCHAR, close DOUBLE)")
    con.execute(
        "CREATE TABLE reddit_sentiment_daily (date DATE, ticker VARCHAR, sentiment_weighted DOUBLE, posts INTEGER)"
    )

    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    price_rows = [(d.date(), "ABC", 100 + i) for i, d in enumerate(dates)]
    sent_rows = [(d.date(), "ABC", (-1) ** i * 0.5, 3) for i, d in enumerate(dates)]

    con.executemany("INSERT INTO prices VALUES (?, ?, ?)", price_rows)
    con.executemany("INSERT INTO reddit_sentiment_daily VALUES (?, ?, ?, ?)", sent_rows)

    result = compute_price_sentiment_correlations(con, ["ABC"], min_samples=4)
    assert "ABC" in result
    entry = result["ABC"]
    assert entry["samples"] >= 4
    assert entry["pearson"] is not None
    assert entry["spearman"] is not None


def test_compute_price_sentiment_correlations_handles_missing_tables(tmp_path):
    con = duckdb.connect(str(tmp_path / "empty.duckdb"))
    result = compute_price_sentiment_correlations(con, ["XYZ"])
    assert result == {}

