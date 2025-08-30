from __future__ import annotations

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import duckdb

from .config import settings

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

    # Backward-compatible aliases
    @property
    def symbol(self) -> str:  # pragma: no cover - simple alias
        return self.ticker

    @property
    def operator(self) -> str:  # pragma: no cover - simple alias
        return self.op


_ALERTS: List[Alert] = []
_NEXT_ID = 1


def _normalize_ticker(ticker: str) -> str:
    return ticker.upper()


def _validate_op(op: str) -> str:
    if op not in _ALLOWED_OPERATORS:
        raise ValueError(f"Invalid operator: {op}")
    return op


def add_alert(ticker: str, op: str, price: float) -> Alert:
    """Add a new alert and return the created object."""
    global _NEXT_ID
    ticker = _normalize_ticker(ticker)
    op = _validate_op(op)
    for i, existing in enumerate(_ALERTS):
        if existing.ticker == ticker:
            alert = Alert(id=existing.id, ticker=ticker, op=op, price=float(price))
            _ALERTS[i] = alert
            return alert
    alert = Alert(id=_NEXT_ID, ticker=ticker, op=op, price=float(price))
    _ALERTS.append(alert)
    _NEXT_ID += 1
    return alert


def list_alerts(ticker: Optional[str] = None) -> List[Alert]:
    if ticker is not None:
        sym = _normalize_ticker(ticker)
        return [a for a in _ALERTS if a.ticker == sym]
    return list(_ALERTS)


def delete_alert(alert_id: Union[int, Alert]) -> bool:
    aid = alert_id.id if isinstance(alert_id, Alert) else int(alert_id)
    for i, alert in enumerate(_ALERTS):
        if alert.id == aid:
            del _ALERTS[i]
            return True
    return False


def active_alerts(ticker: Optional[str] = None) -> List[Alert]:
    if ticker is not None:
        sym = _normalize_ticker(ticker)
        return [a for a in _ALERTS if a.active and a.ticker == sym]
    return [a for a in _ALERTS if a.active]


def deactivate(alert_id: Union[int, Alert]) -> bool:
    aid = alert_id.id if isinstance(alert_id, Alert) else int(alert_id)
    for i, alert in enumerate(_ALERTS):
        if alert.id == aid:
            _ALERTS[i] = Alert(
                id=alert.id,
                ticker=alert.ticker,
                op=alert.op,
                price=alert.price,
                active=False,
            )
            return True
    return False


def activate(alert_id: Union[int, Alert]) -> bool:
    aid = alert_id.id if isinstance(alert_id, Alert) else int(alert_id)
    for i, alert in enumerate(_ALERTS):
        if alert.id == aid:
            _ALERTS[i] = Alert(
                id=alert.id,
                ticker=alert.ticker,
                op=alert.op,
                price=alert.price,
                active=True,
            )
            return True
    return False


# Backwards compatible names
deactivate_alert = deactivate
activate_alert = activate


# -- Optional DB helpers --


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


def add_alert_db(ticker: str, op: str, price: float, db_path: str | None = None) -> int:
    db_path = db_path or settings.WALLENSTEIN_DB_PATH
    with duckdb.connect(db_path) as con:
        _ensure_table(con)
        cur = con.execute(
            "INSERT INTO alerts (ticker, op, price, active) VALUES (?, ?, ?, TRUE)",
            [ticker, op, price],
        )
        row = cur.fetchone()
        return row[0] if row else 0

