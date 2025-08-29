import duckdb

from wallenstein import alerts, watchlist


class DummyConn:
    def __init__(self):
        self._con = duckdb.connect(":memory:")

    def execute(self, *args, **kwargs):
        return self._con.execute(*args, **kwargs)

    def close(self) -> None:  # pragma: no cover - no-op for tests
        pass


def _setup_db(monkeypatch):
    con = DummyConn()
    monkeypatch.setattr(watchlist.duckdb, "connect", lambda _: con)
    monkeypatch.setattr(alerts.duckdb, "connect", lambda _: con)
    return con


def test_watchlist_add_remove_list(monkeypatch):
    _setup_db(monkeypatch)
    watchlist.add_ticker("nvda")
    watchlist.add_ticker("amzn")
    assert watchlist.list_tickers() == ["AMZN", "NVDA"]
    watchlist.remove_ticker("nvda")
    assert watchlist.list_tickers() == ["AMZN"]


def test_alerts_crud(monkeypatch):
    _setup_db(monkeypatch)
    alert_id = alerts.add_alert("NVDA", ">", 100)
    alerts.add_alert("AMZN", "<", 50)
    all_alerts = alerts.list_alerts()
    assert {a.ticker for a in all_alerts} == {"NVDA", "AMZN"}
    alerts.deactivate_alert(alert_id)
    assert [a for a in alerts.list_alerts() if a.id == alert_id][0].active is False
    alerts.activate_alert(alert_id)
    assert [a for a in alerts.list_alerts() if a.id == alert_id][0].active is True
    alerts.delete_alert(alert_id)
    remaining = alerts.list_alerts()
    assert len(remaining) == 1 and remaining[0].ticker == "AMZN"
