from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import duckdb

try:  # optional during some tests
    from .config import settings  # type: ignore
except Exception:  # pragma: no cover - config is optional
    settings = None  # type: ignore

_ALLOWED_OPERATORS = {"<", ">", "<=", ">="}


@dataclass(frozen=True)
class Alert:
    id: int
    ticker: str
    op: str
    price: float
    active: bool = True

    def __hash__(self) -> int:  # allow comparison with ints in sets
        return hash(self.id)

    def __eq__(self, other: object) -> bool:  # pragma: no cover - simple
        if isinstance(other, Alert):
            return self.id == other.id
        if isinstance(other, int):
            return self.id == other
        return NotImplemented

    # Backwards compatible aliases
    @property
    def symbol(self) -> str:  # pragma: no cover - simple alias
        return self.ticker

    @property
    def operator(self) -> str:  # pragma: no cover - simple alias
        return self.op


# ---------- Helpers ----------


def _normalize_ticker(ticker: str) -> str:
    return ticker.upper()


def _validate_op(op: str) -> str:
    if op not in _ALLOWED_OPERATORS:
        raise ValueError(f"Invalid operator: {op}")
    return op


def _get_db_path(db_path: Optional[str] = None) -> str:
    if db_path:
        return db_path
    if settings is None or not getattr(settings, "WALLENSTEIN_DB_PATH", None):
        return "data/wallenstein.duckdb"
    return settings.WALLENSTEIN_DB_PATH


def _ensure_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY,
            ticker VARCHAR,
            op VARCHAR,
            price DOUBLE,
            active BOOLEAN
        )
        """
    )


# ---------- CRUD API ----------


def add_alert(ticker: str, op: str, price: float, db_path: Optional[str] = None) -> Alert:
    """Add a new alert or update existing one for ticker."""
    ticker = _normalize_ticker(ticker)
    op = _validate_op(op)
    path = _get_db_path(db_path)
    con = duckdb.connect(path)
    try:
        _ensure_table(con)
        row = con.execute("SELECT id FROM alerts WHERE ticker = ?", [ticker]).fetchone()
        if row:
            alert_id = int(row[0])
            con.execute(
                "UPDATE alerts SET op = ?, price = ?, active = TRUE WHERE id = ?",
                [op, float(price), alert_id],
            )
        else:
            alert_id = con.execute(
                "SELECT COALESCE(MAX(id), 0) + 1 FROM alerts"
            ).fetchone()[0]
            con.execute(
                "INSERT INTO alerts (id, ticker, op, price, active) VALUES (?, ?, ?, ?, TRUE)",
                [alert_id, ticker, op, float(price)],
            )
        data = con.execute(
            "SELECT id, ticker, op, price, active FROM alerts WHERE id = ?", [alert_id]
        ).fetchone()
        return Alert(*data)
    finally:
        con.close()


def list_alerts(ticker: Optional[str] = None, db_path: Optional[str] = None) -> List[Alert]:
    path = _get_db_path(db_path)
    con = duckdb.connect(path)
    try:
        _ensure_table(con)
        if ticker is not None:
            sym = _normalize_ticker(ticker)
            rows = con.execute(
                "SELECT id, ticker, op, price, active FROM alerts WHERE ticker = ? ORDER BY id",
                [sym],
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT id, ticker, op, price, active FROM alerts ORDER BY id"
            ).fetchall()
        return [Alert(*r) for r in rows]
    finally:
        con.close()


def delete_alert(alert_id: Union[int, Alert], db_path: Optional[str] = None) -> bool:
    aid = alert_id.id if isinstance(alert_id, Alert) else int(alert_id)
    path = _get_db_path(db_path)
    con = duckdb.connect(path)
    try:
        _ensure_table(con)
        row = con.execute(
            "DELETE FROM alerts WHERE id = ? RETURNING id",
            [aid],
        ).fetchone()
        return row is not None
    finally:
        con.close()


def active_alerts(ticker: Optional[str] = None, db_path: Optional[str] = None) -> List[Alert]:
    path = _get_db_path(db_path)
    con = duckdb.connect(path)
    try:
        _ensure_table(con)
        if ticker is not None:
            sym = _normalize_ticker(ticker)
            rows = con.execute(
                "SELECT id, ticker, op, price, active FROM alerts WHERE active AND ticker = ? ORDER BY id",
                [sym],
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT id, ticker, op, price, active FROM alerts WHERE active ORDER BY id"
            ).fetchall()
        return [Alert(*r) for r in rows]
    finally:
        con.close()


def deactivate(alert_id: Union[int, Alert], db_path: Optional[str] = None) -> bool:
    aid = alert_id.id if isinstance(alert_id, Alert) else int(alert_id)
    path = _get_db_path(db_path)
    con = duckdb.connect(path)
    try:
        _ensure_table(con)
        row = con.execute(
            "UPDATE alerts SET active = FALSE WHERE id = ? RETURNING id",
            [aid],
        ).fetchone()
        return row is not None
    finally:
        con.close()


def activate(alert_id: Union[int, Alert], db_path: Optional[str] = None) -> bool:
    aid = alert_id.id if isinstance(alert_id, Alert) else int(alert_id)
    path = _get_db_path(db_path)
    con = duckdb.connect(path)
    try:
        _ensure_table(con)
        row = con.execute(
            "UPDATE alerts SET active = TRUE WHERE id = ? RETURNING id",
            [aid],
        ).fetchone()
        return row is not None
    finally:
        con.close()


# Backwards compatible names

deactivate_alert = deactivate
activate_alert = activate


__all__ = [
    "Alert",
    "add_alert",
    "list_alerts",
    "delete_alert",
    "active_alerts",
    "deactivate",
    "activate",
    "deactivate_alert",
    "activate_alert",
]
