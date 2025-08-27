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
os.environ.setdefault('REDDIT_CLIENT_ID', 'x')
os.environ.setdefault('REDDIT_CLIENT_SECRET', 'x')
os.environ.setdefault('REDDIT_USER_AGENT', 'x')
os.environ.setdefault('TELEGRAM_BOT_TOKEN', 'x')
os.environ.setdefault('TELEGRAM_CHAT_ID', 'x')

import importlib
import wallenstein.config as config_module
importlib.reload(config_module)

from telegram_bot import handle_ticker


class DummyMessage:
    def __init__(self, text: str):
        self.text = text
        self.replies = []

    async def reply_text(self, text: str) -> None:
        self.replies.append(text)


def test_handle_ticker(monkeypatch):
    called = {}

    def fake_run_pipeline(tickers):
        called['pipeline'] = tickers

    def fake_overview(tickers):
        called['overview'] = tickers
        return 'OVERVIEW'

    async def fake_run_in_executor(executor, func, tickers):
        func(tickers)

    app = types.SimpleNamespace(run_in_executor=fake_run_in_executor)

    monkeypatch.setattr('telegram_bot.run_pipeline', fake_run_pipeline)
    monkeypatch.setattr('telegram_bot.generate_overview', fake_overview)

    update = types.SimpleNamespace(message=DummyMessage('!nvda'))
    context = types.SimpleNamespace(application=app)
    asyncio.run(handle_ticker(update, context))
    assert called['pipeline'] == ['NVDA']
    assert called['overview'] == ['NVDA']
    assert update.message.replies == ['OVERVIEW']
