import asyncio
import os
import sys
import types

# Provide dummy telegram modules so that telegram_bot can be imported
sys.modules.setdefault('telegram', types.SimpleNamespace(Update=object))
sys.modules.setdefault(
    'telegram.ext',
    types.SimpleNamespace(
        ApplicationBuilder=object,
        ContextTypes=types.SimpleNamespace(DEFAULT_TYPE=object),
        MessageHandler=object,
        filters=types.SimpleNamespace(TEXT=None, COMMAND=None),
    ),
)

# Ensure required environment variables exist
os.environ.setdefault('CLIENT_ID', 'x')
os.environ.setdefault('CLIENT_SECRET', 'x')
os.environ.setdefault('TELEGRAM_BOT_TOKEN', 'x')

from telegram_bot import handle_ticker


class DummyMessage:
    def __init__(self, text: str):
        self.text = text
        self.replies = []

    async def reply_text(self, text: str) -> None:
        self.replies.append(text)


def test_handle_ticker(monkeypatch):
    called = {}

    def fake_overview(tickers):
        called['tickers'] = tickers
        return 'OVERVIEW'

    monkeypatch.setattr('telegram_bot.generate_overview', fake_overview)
    update = types.SimpleNamespace(message=DummyMessage('!nvda'))
    context = types.SimpleNamespace()
    asyncio.run(handle_ticker(update, context))
    assert called['tickers'] == ['NVDA']
    assert update.message.replies == ['OVERVIEW']
