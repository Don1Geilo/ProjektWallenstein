"""Utility helpers around the user's watchlist.

This is a tiny shim for the purposes of this kata.  In a real
application the watchlist would likely be stored in a database or be
configurable through an API.  For now we simply read the comma separated
list from the configuration and return the unique, normalised symbols.
"""

from __future__ import annotations

from typing import List

from .config import settings


def all_unique_symbols() -> List[str]:
    """Return a list of unique ticker symbols from the configuration.

    ``settings.WALLENSTEIN_TICKERS`` contains a comma separated list of
    tickers.  This helper normalises the values and returns them as a
    list.  It acts as a lightweight replacement for a potential database
    backed watchlist module.
    """

    raw = settings.WALLENSTEIN_TICKERS
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


__all__ = ["all_unique_symbols"]

