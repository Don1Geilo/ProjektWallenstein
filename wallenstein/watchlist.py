"""
Utility helpers to manage a (chat-scoped) watchlist in DuckDB.

- Primary API (recommended): add_symbols/remove_symbols/list_symbols/all_unique_symbols
  operating on a unified table: watchlist(chat_id, symbol, note, UNIQUE(chat_id, symbol)).

- Convenience API (legacy/global use): add_ticker/remove_ticker/list_tickers uses a
  synthetic chat_id "_GLOBAL_" to emulate a single global watchlist.

Configuration:
- settings.WALLENSTEIN_DB_PATH is used by the convenience API only.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple, Union

import duckdb

try:
    # Optional project config; only needed by convenience API.
    from .config import settings  # type: ignore
except Exception:  # pragma: no cover
    settings = None  # type: ignore


# ---------- Schema helpers ----------

def _ensure_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS watchlist (
            chat_id VARCHAR NOT NULL,
            symbol  VARCHAR NOT NULL,
            note    VARCHAR,
            UNIQUE(chat_id, symbol)
        )
        """
    )


# ---------- Core (chat-scoped) API ----------

ChatId = Union[str, int]


def add_symbols(
    con: duckdb.DuckDBPyConnection,
    chat_id: ChatId,
    symbols: Iterable[str],
    note: Optional[str] = None,
) -> List[str]:
    """Add symbols for a given chat_id. Returns normalized (uppercased) list."""
    _ensure_table(con)
    norm = [s.strip().upper() for s in symbols if s and s.strip()]
    if not norm:
        return []
    data = [(str(chat_id), sym, note) for sym in norm]
    if hasattr(con, "executemany"):
        con.executemany(
            "INSERT OR REPLACE INTO watchlist (chat_id, symbol, note) VALUES (?, ?, ?)",
            data,
        )
    else:  # pragma: no cover - fallback for simple test stubs
        for params in data:
            con.execute(
                "INSERT OR REPLACE INTO watchlist (chat_id, symbol, note) VALUES (?, ?, ?)",
                params,
            )
    return norm


def remove_symbols(
    con: duckdb.DuckDBPyConnection,
    chat_id: ChatId,
    symbols: Iterable[str],
) -> int:
    """Remove symbols for chat_id. Returns number of rows removed."""
    _ensure_table(con)
    norm = [s.strip().upper() for s in symbols if s and s.strip()]
    if not norm:
        return 0
    placeholders = ", ".join(["?"] * len(norm))
    params = [str(chat_id), *norm]
    # Count first (DuckDB doesn't return rowcount reliably for DELETE)
    count = con.execute(
        f"SELECT COUNT(*) FROM watchlist WHERE chat_id = ? AND symbol IN ({placeholders})",
        params,
    ).fetchone()[0]
    con.execute(
        f"DELETE FROM watchlist WHERE chat_id = ? AND symbol IN ({placeholders})",
        params,
    )
    return int(count)


def list_symbols(
    con: duckdb.DuckDBPyConnection,
    chat_id: ChatId,
) -> List[Tuple[str, Optional[str]]]:
    """List all (symbol, note) for chat_id, sorted by symbol."""
    _ensure_table(con)
    rows = con.execute(
        "SELECT symbol, note FROM watchlist WHERE chat_id = ? ORDER BY symbol",
        [str(chat_id)],
    ).fetchall()
    return [(row[0], row[1]) for row in rows]


def all_unique_symbols(con: duckdb.DuckDBPyConnection) -> List[str]:
    """Return all unique symbols across all chat_ids (distinct)."""
    _ensure_table(con)
    rows = con.execute(
        "SELECT DISTINCT symbol FROM watchlist ORDER BY symbol"
    ).fetchall()
    return [r[0] for r in rows]


# ---------- Convenience (global) API ----------
# Uses a synthetic "_GLOBAL_" chat to mimic a repo-wide watchlist without Telegram context.

_GLOBAL_CHAT_ID = "_GLOBAL_"


def _get_db_path() -> str:
    if settings is None or not getattr(settings, "WALLENSTEIN_DB_PATH", None):
        return "data/wallenstein.duckdb"
    return settings.WALLENSTEIN_DB_PATH


def add_ticker(ticker: str, db_path: Optional[str] = None) -> None:
    """Add a single ticker to the global watchlist."""
    path = db_path or _get_db_path()
    con = duckdb.connect(path)
    try:
        add_symbols(con, _GLOBAL_CHAT_ID, [ticker])
    finally:
        con.close()


def remove_ticker(ticker: str, db_path: Optional[str] = None) -> None:
    """Remove a single ticker from the global watchlist."""
    path = db_path or _get_db_path()
    con = duckdb.connect(path)
    try:
        remove_symbols(con, _GLOBAL_CHAT_ID, [ticker])
    finally:
        con.close()


def list_tickers(db_path: Optional[str] = None) -> List[str]:
    """List all tickers in the global watchlist."""
    path = db_path or _get_db_path()
    con = duckdb.connect(path)
    try:
        return [sym for sym, _ in list_symbols(con, _GLOBAL_CHAT_ID)]
    finally:
        con.close()


__all__ = [
    # core
    "add_symbols",
    "remove_symbols",
    "list_symbols",
    "all_unique_symbols",
    # convenience
    "add_ticker",
    "remove_ticker",
    "list_tickers",
]
