from wallenstein.notify import notify_telegram


class DummyResponse:
    ok = True

    @staticmethod
    def json():
        return {"ok": True}


def test_notify_telegram_without_reddit_credentials(monkeypatch):
    """notify_telegram should work without Reddit env vars."""
    monkeypatch.delenv("CLIENT_ID", raising=False)
    monkeypatch.delenv("CLIENT_SECRET", raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")

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
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    assert notify_telegram("hi") is False
