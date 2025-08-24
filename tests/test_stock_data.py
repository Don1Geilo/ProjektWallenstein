import logging

import duckdb
import pandas as pd
import pytest

from wallenstein import stock_data


def test_update_prices_fetches_full_history_for_new_ticker(monkeypatch, tmp_path, caplog):
    db = tmp_path / "prices.duckdb"
    called = {}

    def fake_fetch_one(ticker, start=None):
        called["start"] = start
        return pd.DataFrame({
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "ticker": [ticker, ticker],
            "open": [1.0, 1.1],
            "high": [1.0, 1.1],
            "low": [1.0, 1.1],
            "close": [1.0, 1.1],
            "adj_close": [1.0, 1.1],
            "volume": [100, 200],
        })

    monkeypatch.setattr(stock_data, "_stooq_fetch_one", fake_fetch_one)
    stock_data.DATA_SOURCE = "stooq"

    caplog.set_level(logging.WARNING)
    n = stock_data.update_prices(str(db), ["NEW"])
    assert n == 2
    assert called["start"] is None
    assert "no trading data" not in caplog.text


def test_update_prices_skips_weekend(monkeypatch, tmp_path, caplog):
    db = tmp_path / "prices.duckdb"
    con = duckdb.connect(str(db))
    con.execute(
        "CREATE TABLE prices (date DATE, ticker VARCHAR, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, adj_close DOUBLE, volume BIGINT)"
    )
    # last trading day Friday 2024-04-05
    con.execute(
        "INSERT INTO prices VALUES ('2024-04-05','NVDA',1,1,1,1,1,1)"
    )
    con.close()

    called = {"stooq": 0}

    def fake_fetch_one(ticker, start=None):
        called["stooq"] += 1
        return pd.DataFrame()

    monkeypatch.setattr(stock_data, "_stooq_fetch_one", fake_fetch_one)
    stock_data.DATA_SOURCE = "stooq"

    monkeypatch.setattr(pd.Timestamp, "utcnow", lambda: pd.Timestamp("2024-04-06"))
    caplog.set_level(logging.INFO)
    n = stock_data.update_prices(str(db), ["NVDA"])
    assert n == 0
    assert called["stooq"] == 0
    assert "no trading data" not in caplog.text
