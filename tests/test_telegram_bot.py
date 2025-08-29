import asyncio
import os
import sys
import types
import importlib

# Provide dummy telegram modules so that telegram_bot can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
class DummyApp:
    @classmethod
    def builder(cls):
        class Builder:
            def token(self, token):
                return self

            def build(self):
                return DummyApp()

        return Builder()

    def add_handler(self, handler):
        pass

    def run_polling(self):
        pass

sys.modules.setdefault("telegram", types.SimpleNamespace(Update=object))
sys.modules.setdefault(
    "telegram.ext",
    types.SimpleNamespace(
        Application=DummyApp,
        CommandHandler=object,
        ContextTypes=types.SimpleNamespace(DEFAULT_TYPE=object),
    ),
)

# Ensure required environment variables exist
os.environ.setdefault("REDDIT_CLIENT_ID", "x")
os.environ.setdefault("REDDIT_CLIENT_SECRET", "x")
os.environ.setdefault("REDDIT_USER_AGENT", "x")
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "x")
os.environ.setdefault("TELEGRAM_CHAT_ID", "x")

# Reload config to pick up env vars
import wallenstein.config as config_module
importlib.reload(config_module)

from bot import telegram_bot as bot


class DummyMessage:
    def __init__(self):
        self.replies: list[str] = []

    async def reply_text(self, text: str, **kwargs) -> None:
        self.replies.append(text)


def run(coro):
    asyncio.get_event_loop().run_until_complete(coro)


def test_add(monkeypatch):
    called = {}

    def fake_add(symbol):
        called["symbol"] = symbol

    monkeypatch.setattr(bot.watchlist, "add", fake_add)

    update = types.SimpleNamespace(message=DummyMessage())
    context = types.SimpleNamespace(args=["nvda"])
    run(bot.add(update, context))
    assert called["symbol"] == "NVDA"
    assert "NVDA" in update.message.replies[0]


def test_list(monkeypatch):
    monkeypatch.setattr(bot.watchlist, "list_symbols", lambda: ["AAPL", "TSLA"])
    update = types.SimpleNamespace(message=DummyMessage())
    context = types.SimpleNamespace(args=[])
    run(bot.list_symbols(update, context))
    assert "AAPL" in update.message.replies[0]
    assert "TSLA" in update.message.replies[0]


def test_alerts_add(monkeypatch):
    called = {}

    def fake_add(symbol, op, price):
        called["args"] = (symbol, op, price)
        return 1

    monkeypatch.setattr(bot.alerts, "add", fake_add)

    update = types.SimpleNamespace(message=DummyMessage())
    context = types.SimpleNamespace(args=["add", "tsla", "<", "100"])
    run(bot.alerts_command(update, context))
    assert called["args"] == ("TSLA", "<", 100.0)
    assert "TSLA" in update.message.replies[0]


def test_alerts_del(monkeypatch):
    called = {}

    def fake_del(alert_id):
        called["id"] = alert_id
        return True

    monkeypatch.setattr(bot.alerts, "delete", fake_del)

    update = types.SimpleNamespace(message=DummyMessage())
    context = types.SimpleNamespace(args=["del", "3"])
    run(bot.alerts_command(update, context))
    assert called["id"] == 3
    assert "3" in update.message.replies[0]
