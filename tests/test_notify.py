from wallenstein.notify import TELEGRAM_MAX_LENGTH, notify_telegram


class DummyResponse:
    ok = True

    @staticmethod
    def json():
        return {"ok": True}


def test_notify_telegram_without_reddit_credentials(monkeypatch):
    """notify_telegram should work without Reddit env vars."""
    monkeypatch.delenv("CLIENT_ID", raising=False)
    monkeypatch.delenv("CLIENT_SECRET", raising=False)
    monkeypatch.setattr("wallenstein.notify.settings.TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setattr("wallenstein.notify.settings.TELEGRAM_CHAT_ID", "123")

    called = {}

    def fake_post(url, data):
        called["url"] = url
        called["data"] = data
        return DummyResponse()

    monkeypatch.setattr("wallenstein.notify.requests.post", fake_post)

    assert notify_telegram("hi there") is True
    assert called["url"].endswith("bottoken/sendMessage")
    assert called["data"]["chat_id"] == "123"


def test_notify_telegram_missing_config(monkeypatch):
    monkeypatch.setattr("wallenstein.notify.settings.TELEGRAM_BOT_TOKEN", None)
    monkeypatch.setattr("wallenstein.notify.settings.TELEGRAM_CHAT_ID", None)
    assert notify_telegram("hi") is False


def test_notify_telegram_splits_long_messages(monkeypatch):
    monkeypatch.setattr("wallenstein.notify.settings.TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setattr("wallenstein.notify.settings.TELEGRAM_CHAT_ID", "123")

    calls = []

    class MultiResponse:
        ok = True

        @staticmethod
        def json():
            return {"ok": True}

    def fake_post(url, data):
        calls.append(data["text"])
        return MultiResponse()

    monkeypatch.setattr("wallenstein.notify.requests.post", fake_post)

    text = "A" * (TELEGRAM_MAX_LENGTH + 50)
    assert notify_telegram(text) is True
    assert len(calls) == 2
    assert all(len(chunk) <= TELEGRAM_MAX_LENGTH for chunk in calls)
