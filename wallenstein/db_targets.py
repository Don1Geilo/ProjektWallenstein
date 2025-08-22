# wallenstein/db_targets.py
from __future__ import annotations
import duckdb
from typing import List, Dict, Any

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS broker_targets (
  ticker TEXT,
  fetched_at_utc BIGINT,
  target_mean DOUBLE,
  target_high DOUBLE,
  target_low DOUBLE,
  rec_mean DOUBLE,
  rec_text TEXT,
  strong_buy INTEGER,
  buy INTEGER,
  hold INTEGER,
  sell INTEGER,
  strong_sell INTEGER,
  source TEXT,
  PRIMARY KEY (ticker, fetched_at_utc)
);
"""

_UPSERT_SQL = """
INSERT OR REPLACE INTO broker_targets
(ticker, fetched_at_utc, target_mean, target_high, target_low,
 rec_mean, rec_text, strong_buy, buy, hold, sell, strong_sell, source)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

def init(db_path: str) -> None:
    """Erzeugt – falls nötig – die Tabelle broker_targets in der DuckDB."""
    con = duckdb.connect(db_path)
    try:
        con.execute(_SCHEMA_SQL)
    finally:
        con.close()

def save_snapshots(db_path: str, rows: List[Dict[str, Any]]) -> None:
    """Schreibt/ersetzt Snapshots in broker_targets (PRIMARY KEY = ticker+timestamp)."""
    if not rows:
        return
    con = duckdb.connect(db_path)
    try:
        con.execute(_SCHEMA_SQL)
        for r in rows:
            con.execute(
                _UPSERT_SQL,
                [
                    r.get("ticker"),
                    r.get("fetched_at_utc"),
                    r.get("target_mean"),
                    r.get("target_high"),
                    r.get("target_low"),
                    r.get("rec_mean"),
                    r.get("rec_text"),
                    r.get("strong_buy"),
                    r.get("buy"),
                    r.get("hold"),
                    r.get("sell"),
                    r.get("strong_sell"),
                    r.get("source"),
                ],
            )
    finally:
        con.close()