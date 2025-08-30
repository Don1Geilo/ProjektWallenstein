import duckdb
import pytest

from wallenstein import alerts


class DummyConn:
    def __init__(self):
        self._con = duckdb.connect(":memory:")

    def execute(self, *args, **kwargs):
        return self._con.execute(*args, **kwargs)

    def close(self) -> None:  # pragma: no cover - no-op for tests
        pass


def _setup_db(monkeypatch):
    con = DummyConn()
    monkeypatch.setattr(alerts.duckdb, "connect", lambda _: con)
    return con


def test_add_list_delete_alert(monkeypatch):
    _setup_db(monkeypatch)
    a1 = alerts.add_alert("aapl", "<=", 100)
    assert a1.symbol == "AAPL"
    assert a1.operator == "<="
    assert len(alerts.list_alerts()) == 1

    assert alerts.delete_alert(a1.id) is True
    assert alerts.list_alerts() == []


def test_invalid_operator(monkeypatch):
    _setup_db(monkeypatch)
    with pytest.raises(ValueError):
        alerts.add_alert("aapl", "==", 100)


def test_activation_deactivation_flow(monkeypatch):
    _setup_db(monkeypatch)
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


def test_reactivate_alert(monkeypatch):
    _setup_db(monkeypatch)
    first = alerts.add_alert("NVDA", ">", 100)
    second = alerts.add_alert("AMZN", "<", 50)
    assert {a.id for a in alerts.active_alerts()} == {first, second}

    alerts.deactivate(first)
    assert {a.id for a in alerts.active_alerts()} == {second}

    assert alerts.activate(first) is True
    assert {a.id for a in alerts.active_alerts()} == {first, second}
