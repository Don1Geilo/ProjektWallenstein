"""Simple price alert persistence API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

_DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "alerts.json"


def _load() -> List[Dict]:
    try:
        return json.loads(_DATA_FILE.read_text())
    except FileNotFoundError:
        return []


def _save(data: List[Dict]) -> None:
    _DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    _DATA_FILE.write_text(json.dumps(data))


def add(symbol: str, op: str, price: float) -> int:
    """Add an alert and return its new ID."""
    data = _load()
    next_id = max((a.get("id", 0) for a in data), default=0) + 1
    alert = {"id": next_id, "symbol": symbol, "op": op, "price": price}
    data.append(alert)
    _save(data)
    return next_id


def list_alerts() -> List[Dict]:
    """Return all alerts."""
    return _load()


def delete(alert_id: int) -> bool:
    """Delete alert by ID. Return True if removed."""
    data = _load()
    for alert in list(data):
        if int(alert.get("id")) == int(alert_id):
            data.remove(alert)
            _save(data)
            return True
    return False
