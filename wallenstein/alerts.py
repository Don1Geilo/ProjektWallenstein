"""In-memory alert management utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

_ALLOWED_OPERATORS = {"<", ">", "<=", ">="}


@dataclass
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
