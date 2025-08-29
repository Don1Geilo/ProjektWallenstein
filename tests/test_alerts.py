import duckdb

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


def test_reactivate_alert(monkeypatch):
    _setup_db(monkeypatch)
    first = alerts.add_alert("NVDA", ">", 100)
    second = alerts.add_alert("AMZN", "<", 50)
    assert {a.id for a in alerts.active_alerts()} == {first, second}

    alerts.deactivate(first)
    assert {a.id for a in alerts.active_alerts()} == {second}

    assert alerts.activate(first) is True
    assert {a.id for a in alerts.active_alerts()} == {first, second}
