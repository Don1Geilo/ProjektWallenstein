import duckdb

from .config import settings


def _ensure_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("CREATE TABLE IF NOT EXISTS watchlist (ticker VARCHAR PRIMARY KEY)")


def add_ticker(ticker: str, db_path: str | None = None) -> None:
    db_path = db_path or settings.WALLENSTEIN_DB_PATH
    con = duckdb.connect(db_path)
    try:
        _ensure_table(con)
        con.execute("INSERT OR REPLACE INTO watchlist (ticker) VALUES (?)", [ticker.upper()])
    finally:
        con.close()


def remove_ticker(ticker: str, db_path: str | None = None) -> None:
    db_path = db_path or settings.WALLENSTEIN_DB_PATH
    con = duckdb.connect(db_path)
    try:
        _ensure_table(con)
        con.execute("DELETE FROM watchlist WHERE ticker = ?", [ticker.upper()])
    finally:
        con.close()


def list_tickers(db_path: str | None = None) -> list[str]:
    db_path = db_path or settings.WALLENSTEIN_DB_PATH
    con = duckdb.connect(db_path)
    try:
        _ensure_table(con)
        rows = con.execute("SELECT ticker FROM watchlist ORDER BY ticker").fetchall()
        return [r[0] for r in rows]
    finally:
        con.close()
