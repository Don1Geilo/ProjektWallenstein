"""Simple in-memory alert checking helper.

The real project this kata is derived from exposes an external API for
price alerts.  For the purposes of the exercises we implement a very
small subset locally so that the main pipeline can interact with it
without depending on any network resources.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List
import logging

log = logging.getLogger(__name__)


@dataclass
class Alert:
    id: int
    ticker: str
    threshold: float
    direction: str = "above"  # either "above" or "below"


_ALERTS: List[Alert] = []


def register_alert(alert: Alert) -> None:
    """Register a new alert in the in-memory store."""

    _ALERTS.append(alert)


def active_alerts(prices: Dict[str, float], notifier: Callable[[str], bool] | None = None) -> None:
    """Check active alerts against the provided ``prices``.

    If an alert is triggered the ``notifier`` callable is invoked with a
    human readable message and the alert is removed ("deactivated").
    """

    remaining: List[Alert] = []
    for alert in _ALERTS:
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
    _ALERTS[:] = remaining


__all__ = ["Alert", "active_alerts", "register_alert"]

