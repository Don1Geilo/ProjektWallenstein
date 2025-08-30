"""Database helpers for Wallenstein.

This module provides a small wrapper around DuckDB that ensures the
application's schema is initialised.  The schema itself lives in
``schema.sql`` so it can easily be inspected or adapted without touching
Python code.
"""

from __future__ import annotations

import logging
from importlib import resources
from pathlib import Path

import duckdb

from .config import settings
from .db_schema import ensure_tables


def get_connection(db_path: str | None = None) -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection.

    The directory of ``db_path`` is created automatically.  If ``db_path`` is
    ``None`` the path from the application settings is used.
    """
    path = Path(db_path or settings.WALLENSTEIN_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(path)


def _execute_script(con: duckdb.DuckDBPyConnection, script: str) -> None:
    """Execute a multi-statement SQL script."""
    for statement in script.split(";"):
        stmt = statement.strip()
        if stmt:
            con.execute(stmt)


def init_schema(db_path: str | None = None) -> None:
    """Create required tables if they do not yet exist."""
    with get_connection(db_path) as con:
        schema_sql = resources.files(__package__).joinpath("schema.sql").read_text()
        _execute_script(con, schema_sql)
        ensure_tables(con)
        con.execute(
            """
        CREATE TABLE IF NOT EXISTS ticker_aliases (
          ticker VARCHAR,
          alias  VARCHAR,
          source VARCHAR,
          added_at TIMESTAMP DEFAULT NOW(),
          UNIQUE(ticker, alias)
        )
        """
        )
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_ticker_aliases_alias ON ticker_aliases(alias)"
        )
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_ticker_aliases_tkr ON ticker_aliases(ticker)"
        )

    from wallenstein.aliases import seed_from_json
    try:
        import duckdb
        with duckdb.connect(db_path or settings.WALLENSTEIN_DB_PATH) as con:
            n = seed_from_json(con)
            if n:
                log = logging.getLogger("wallenstein")
                log.info(f"Aliase aus JSON importiert: {n}")
    except Exception as e:  # pragma: no cover - best effort
        logging.getLogger("wallenstein").warning(
            f"Alias-Seed übersprungen: {e}"
        )
