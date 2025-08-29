"""Simple in-memory alert checking helper.

The real project this kata is derived from exposes an external API for
price alerts.  For the purposes of the exercises we implement a very
small subset locally so that the main pipeline can interact with it
without depending on any network resources.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from itertools import count
from typing import Callable, Dict, List

log = logging.getLogger(__name__)


@dataclass
class Alert:
    id: int
    ticker: str
    threshold: float
    direction: str = "above"  # either "above" or "below"


_NEXT_ID = count(1)
_ALERTS: List[Alert] = []
_ALERTS_LOCK = threading.Lock()

def register_alert(ticker: str, threshold: float, direction: str = "above") -> int:
    """Register a new alert in the in-memory store.

    Returns the ID of the newly created alert.
    """

    alert = Alert(next(_NEXT_ID), ticker, threshold, direction)
    with _ALERTS_LOCK:
        _ALERTS.append(alert)
    return alert.id


def active_alerts(prices: Dict[str, float], notifier: Callable[[str], bool] | None = None) -> None:
    """Check active alerts against the provided ``prices``.

    If an alert is triggered the ``notifier`` callable is invoked with a
    human readable message and the alert is removed ("deactivated").
    """

    with _ALERTS_LOCK:
        alerts_snapshot = list(_ALERTS)

    remaining: List[Alert] = []
    for alert in alerts_snapshot:
        price = prices.get(alert.ticker)
        if price is None:
            remaining.append(alert)
            continue
        triggered = (
            price >= alert.threshold if alert.direction == "above" else price <= alert.threshold
        )
        if triggered:
            message = (
                f"{alert.ticker} price {price:.2f} {'≥' if alert.direction == 'above' else '≤'} "
                f"{alert.threshold:.2f}"
            )
            if notifier:
                try:
                    notifier(message)
                except Exception as exc:  # pragma: no cover - notifier failures
                    log.warning("Alert notification failed: %s", exc)
        else:
            remaining.append(alert)

    # deactivate triggered alerts
    with _ALERTS_LOCK:
        _ALERTS[:] = remaining


def list_alerts() -> List[Alert]:
    """Return a copy of all currently registered alerts."""

    with _ALERTS_LOCK:
        return list(_ALERTS)


def _reset_state() -> None:  # pragma: no cover - used only in tests
    """Reset global state for tests."""

    global _NEXT_ID
    with _ALERTS_LOCK:
        _ALERTS.clear()
    _NEXT_ID = count(1)


__all__ = ["Alert", "active_alerts", "register_alert", "list_alerts"]

