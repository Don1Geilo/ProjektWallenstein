import logging
import math
from datetime import date, timedelta

import duckdb
import pandas as pd
import requests

from wallenstein import stock_data


def test_update_prices_fetches_full_history_for_new_ticker(monkeypatch, tmp_path, caplog):
    db = tmp_path / "prices.duckdb"
    called = {}

    def fake_fetch_one(ticker, start=None, session=None):
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

    def fake_fetch_one(ticker, start=None, session=None):
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


def test_stooq_fetch_one_retries(monkeypatch):
    class DummyResp:
        def __init__(self, ok, text):
            self.ok = ok
            self.text = text

    class DummySession:
        def __init__(self):
            self.calls = 0

        def get(self, url, timeout=20, headers=None):
            self.calls += 1
            if self.calls == 1:
                raise requests.exceptions.ConnectionError("boom")
            return DummyResp(True, "Date,Open,High,Low,Close,Volume\n2024-01-02,1,1,1,1,100")

    sess = DummySession()
    df = stock_data._stooq_fetch_one("TEST", session=sess)
    assert sess.calls == 2
    assert not df.empty


def test_download_single_safe_handles_429(monkeypatch, caplog):
    class DummyTicker:
        def history(self, *args, **kwargs):
            resp = requests.Response()
            resp.status_code = 429
            raise requests.exceptions.HTTPError(response=resp)

    monkeypatch.setattr(stock_data.yf, "Ticker", lambda *_, **__: DummyTicker())
    caplog.set_level(logging.WARNING)
    df = stock_data._download_single_safe("FAIL", session=requests.Session())
    assert df.empty
    assert "skipped due to rate limiting" in caplog.text


def test_download_single_safe_falls_back_when_session_rejected(monkeypatch):
    from yfinance.exceptions import YFDataException

    class DummyTicker:
        def history(self, *args, **kwargs):
            return pd.DataFrame(
                {
                    "Open": [1.0],
                    "High": [1.0],
                    "Low": [1.0],
                    "Close": [1.0],
                    "Adj Close": [1.0],
                    "Volume": [100],
                },
                index=pd.to_datetime(["2024-01-02"]),
            )

    def fake_ticker(symbol, session=None):
        if session is not None:
            raise YFDataException("invalid session")
        return DummyTicker()

    monkeypatch.setattr(stock_data, "_yahoo_chart_api_daily", lambda *_, **__: pd.DataFrame())
    monkeypatch.setattr(stock_data.yf, "Ticker", fake_ticker)

    df = stock_data._download_single_safe("TEST", session=requests.Session())
    assert not df.empty
    assert df["ticker"].iloc[0] == "TEST"


def test_prices_table_has_index(tmp_path):
    db = tmp_path / "idx.duckdb"
    con = duckdb.connect(str(db))
    stock_data._ensure_prices_table(con)
    idx_count = con.execute(
        "SELECT COUNT(*) FROM duckdb_indexes() WHERE index_name='prices_ticker_date_idx'"
    ).fetchone()[0]
    con.close()
    assert idx_count == 1


def test_get_last_close_returns_value(tmp_path):
    db = tmp_path / "last.duckdb"
    con = duckdb.connect(str(db))
    stock_data._ensure_prices_table(con)
    con.execute(
        "INSERT INTO prices VALUES ('2024-01-01','AAA',1,1,1,10,10,100)"
    )
    con.execute(
        "INSERT INTO prices VALUES ('2024-01-02','AAA',1,1,1,20,20,200)"
    )
    val = stock_data.get_last_close(con, "AAA")
    con.close()
    assert val == 20


def test_get_last_close_returns_nan_when_missing(tmp_path):
    db = tmp_path / "empty.duckdb"
    con = duckdb.connect(str(db))
    stock_data._ensure_prices_table(con)
    val = stock_data.get_last_close(con, "NONE")
    con.close()
    assert math.isnan(val)


def test_purge_old_prices_removes_expired_rows(tmp_path, monkeypatch):
    db = tmp_path / "purge.duckdb"
    con = duckdb.connect(str(db))
    stock_data._ensure_prices_table(con)
    old_date = date.today() - timedelta(days=40)
    new_date = date.today()
    con.execute(
        "INSERT INTO prices VALUES (?,?,?,?,?,?,?,?)",
        (old_date, "AAA", 1, 1, 1, 1, 1, 1),
    )
    con.execute(
        "INSERT INTO prices VALUES (?,?,?,?,?,?,?,?)",
        (new_date, "AAA", 1, 1, 1, 1, 1, 1),
    )
    con.close()

    monkeypatch.setattr(stock_data, "DATA_RETENTION_DAYS", 30)
    stock_data.purge_old_prices(str(db))

    with duckdb.connect(str(db)) as con:
        rows = con.execute("SELECT date FROM prices ORDER BY date").fetchall()
    assert rows == [(new_date,)]
