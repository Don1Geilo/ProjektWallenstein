"""Simple watchlist persistence API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

_DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "watchlist.json"


def _load() -> List[str]:
    try:
        return json.loads(_DATA_FILE.read_text())
    except FileNotFoundError:
        return []


def _save(data: List[str]) -> None:
    _DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    _DATA_FILE.write_text(json.dumps(data))


def add(symbol: str) -> None:
    """Add ``symbol`` to the watchlist."""
    data = _load()
    if symbol not in data:
        data.append(symbol)
        _save(data)


def remove(symbol: str) -> None:
    """Remove ``symbol`` from the watchlist if present."""
    data = _load()
    if symbol in data:
        data.remove(symbol)
        _save(data)


def list_symbols() -> List[str]:
    """Return all watchlist symbols."""
    return _load()
