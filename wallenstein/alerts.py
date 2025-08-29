
"""In-memory alert management utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

_ALLOWED_OPERATORS = {"<", ">", "<=", ">="}

from dataclasses import dataclass

import duckdb

from .config import settings




class Alert:

    """Simple alert representation."""

    id: int
    symbol: str
    operator: str
    price: float
    active: bool = True


_ALERTS: List[Alert] = []
_NEXT_ID = 1


def _normalize_symbol(symbol: str) -> str:
    return symbol.upper()


def _validate_operator(op: str) -> str:
    if op not in _ALLOWED_OPERATORS:
        raise ValueError(f"Invalid operator: {op}")
    return op


def add_alert(symbol: str, operator: str, price: float) -> Alert:
    """Add a new alert and return it.

    Operator must be one of ``<``, ``>``, ``<=`` or ``>=``.
    The symbol is normalised to uppercase.
    """

    global _NEXT_ID
    symbol = _normalize_symbol(symbol)
    operator = _validate_operator(operator)
    alert = Alert(id=_NEXT_ID, symbol=symbol, operator=operator, price=float(price))
    _ALERTS.append(alert)
    _NEXT_ID += 1
    return alert


def list_alerts(symbol: Optional[str] = None) -> List[Alert]:
    """Return all alerts, optionally filtered by symbol."""

    if symbol is not None:
        symbol = _normalize_symbol(symbol)
        return [a for a in _ALERTS if a.symbol == symbol]
    return list(_ALERTS)


def delete_alert(alert_id: int) -> bool:
    """Delete alert by ID. Returns ``True`` if removed."""

    for i, alert in enumerate(_ALERTS):
        if alert.id == alert_id:
            del _ALERTS[i]
            return True
    return False


def active_alerts(symbol: Optional[str] = None) -> List[Alert]:
    """Return all active alerts, optionally filtered by symbol."""

    if symbol is not None:
        symbol = _normalize_symbol(symbol)
        return [a for a in _ALERTS if a.active and a.symbol == symbol]
    return [a for a in _ALERTS if a.active]


def deactivate(alert_id: int) -> bool:
    """Deactivate an alert by ID. Returns ``True`` if found."""

    for alert in _ALERTS:
        if alert.id == alert_id:
            alert.active = False
            return True
    return False
=======
    id: int
    ticker: str
    op: str
    price: float
    active: bool


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


def add_alert(ticker: str, op: str, price: float, db_path: str | None = None) -> int:
    db_path = db_path or settings.WALLENSTEIN_DB_PATH
    con = duckdb.connect(db_path)
    try:
        _ensure_table(con)
        alert_id = con.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM alerts").fetchone()[0]
        con.execute(
            "INSERT INTO alerts (id, ticker, op, price, active) VALUES (?, ?, ?, ?, TRUE)",
            [alert_id, ticker.upper(), op, float(price)],
        )
        return int(alert_id)
    finally:
        con.close()


def list_alerts(db_path: str | None = None) -> list[Alert]:
    db_path = db_path or settings.WALLENSTEIN_DB_PATH
    con = duckdb.connect(db_path)
    try:
        _ensure_table(con)
        rows = con.execute(
            "SELECT id, ticker, op, price, active FROM alerts ORDER BY id"
        ).fetchall()
        return [Alert(int(r[0]), r[1], r[2], float(r[3]), bool(r[4])) for r in rows]
    finally:
        con.close()


def active_alerts(db_path: str | None = None) -> list[Alert]:
    """Return only alerts marked as active."""

    return [a for a in list_alerts(db_path) if a.active]


def delete_alert(alert_id: int, db_path: str | None = None) -> None:
    db_path = db_path or settings.WALLENSTEIN_DB_PATH
    con = duckdb.connect(db_path)
    try:
        _ensure_table(con)
        con.execute("DELETE FROM alerts WHERE id = ?", [alert_id])
    finally:
        con.close()


def _set_active(alert_id: int, active: bool, db_path: str | None = None) -> bool:
    db_path = db_path or settings.WALLENSTEIN_DB_PATH
    con = duckdb.connect(db_path)
    try:
        _ensure_table(con)
        con.execute("UPDATE alerts SET active = ? WHERE id = ?", [active, alert_id])
        row = con.execute("SELECT COUNT(*) FROM alerts WHERE id = ?", [alert_id]).fetchone()
        return bool(row and row[0])
    finally:
        con.close()


def activate(alert_id: int, db_path: str | None = None) -> bool:
    """Activate the alert with the given ``alert_id``.

    Returns ``True`` if the alert exists and was activated.
    """

    return _set_active(alert_id, True, db_path)


def deactivate(alert_id: int, db_path: str | None = None) -> bool:
    """Deactivate the alert with the given ``alert_id``."""

    return _set_active(alert_id, False, db_path)


# Backwards compatibility
activate_alert = activate
deactivate_alert = deactivate

