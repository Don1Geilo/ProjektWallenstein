import sys
import threading
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import alerts_api


def setup_function(_):
    alerts_api._reset_state()


def test_register_alert_unique_ids():
    def worker():
        alerts_api.register_alert("TST", 1.0)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    alerts = alerts_api.list_alerts()
    ids = [a.id for a in alerts]
    assert len(ids) == 10
    assert len(set(ids)) == 10


def test_active_alerts_removes_triggered():
    alerts_api.register_alert("ABC", 100.0)
    alerts_api.register_alert("XYZ", 50.0, direction="below")
    messages = []

    def notifier(msg: str) -> bool:
        messages.append(msg)
        return True

    alerts_api.active_alerts({"ABC": 150.0, "XYZ": 60.0}, notifier)
    remaining = alerts_api.list_alerts()
    assert [a.ticker for a in remaining] == ["XYZ"]
    assert len(messages) == 1
