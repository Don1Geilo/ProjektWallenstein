
from typing import Iterable, List, Tuple

import duckdb


def _ensure_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS watchlist (
            chat_id VARCHAR,
            symbol VARCHAR,
            note VARCHAR,
            UNIQUE(chat_id, symbol)
        )
        """
    )


def add_symbols(
    con: duckdb.DuckDBPyConnection,
    chat_id: str,
    symbols: Iterable[str],
    note: str | None = None,
) -> List[str]:
    """Add ``symbols`` for ``chat_id``.

    Symbols are uppercased before insertion. Returns the list of inserted symbols
    (uppercased).
    """
    _ensure_table(con)
    symbols_upper = [s.upper() for s in symbols]
    data = [(chat_id, sym, note) for sym in symbols_upper]
    if data:
        con.executemany(
            "INSERT OR REPLACE INTO watchlist (chat_id, symbol, note) VALUES (?, ?, ?)",
            data,
        )
    return symbols_upper


def remove_symbols(
    con: duckdb.DuckDBPyConnection,
    chat_id: str,
    symbols: Iterable[str],
) -> int:
    """Remove ``symbols`` for ``chat_id`` and return number of removed rows."""
    _ensure_table(con)
    symbols_upper = [s.upper() for s in symbols]
    if not symbols_upper:
        return 0
    placeholders = ", ".join(["?"] * len(symbols_upper))
    params = [chat_id, *symbols_upper]
    count = con.execute(
        f"SELECT COUNT(*) FROM watchlist WHERE chat_id = ? AND symbol IN ({placeholders})",
        params,
    ).fetchone()[0]
    con.execute(
        f"DELETE FROM watchlist WHERE chat_id = ? AND symbol IN ({placeholders})",
        params,
    )
    return int(count)


def list_symbols(con: duckdb.DuckDBPyConnection, chat_id: str) -> List[Tuple[str, str | None]]:
    """List all symbols with optional note for ``chat_id``."""
    _ensure_table(con)
    rows = con.execute(
        "SELECT symbol, note FROM watchlist WHERE chat_id = ? ORDER BY symbol",
        [chat_id],
    ).fetchall()
    return [(row[0], row[1]) for row in rows]


def all_unique_symbols(con: duckdb.DuckDBPyConnection) -> List[str]:
    """Return all unique symbols across all chat ids."""
    _ensure_table(con)
    rows = con.execute("SELECT DISTINCT symbol FROM watchlist ORDER BY symbol").fetchall()
    return [row[0] for row in rows]
=======
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

