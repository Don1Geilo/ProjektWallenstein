import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wallenstein import alerts


def setup_function():
    alerts._ALERTS.clear()
    alerts._NEXT_ID = 1


def test_add_list_delete_alert():
    a1 = alerts.add_alert("aapl", "<=", 100)
    assert a1.symbol == "AAPL"
    assert a1.operator == "<="
    assert len(alerts.list_alerts()) == 1

    assert alerts.delete_alert(a1.id) is True
    assert alerts.list_alerts() == []


def test_invalid_operator():
    with pytest.raises(ValueError):
        alerts.add_alert("aapl", "==", 100)


def test_activation_deactivation_flow():
    a1 = alerts.add_alert("msft", ">=", 200)
    a2 = alerts.add_alert("goog", "<", 150)

    assert len(alerts.active_alerts()) == 2

    assert alerts.deactivate(a1.id) is True

    actives = alerts.active_alerts()
    assert len(actives) == 1
    assert actives[0].id == a2.id

    all_alerts = alerts.list_alerts()
    assert len(all_alerts) == 2
    assert not [a for a in all_alerts if a.id == a1.id][0].active
