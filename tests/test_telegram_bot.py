import asyncio
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
        CommandHandler=object,
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

from telegram_bot import cmd_add, cmd_alerts, cmd_list, cmd_remove, handle_ticker


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


def test_cmd_add(monkeypatch):
    called = {}

    def fake_add(ticker):
        called['ticker'] = ticker

    monkeypatch.setattr('telegram_bot.add_ticker', fake_add)
    update = types.SimpleNamespace(message=DummyMessage(''))
    context = types.SimpleNamespace(args=['nvda'])
    asyncio.run(cmd_add(update, context))
    assert called['ticker'] == 'NVDA'
    assert update.message.replies == ['NVDA added.']


def test_cmd_remove(monkeypatch):
    called = {}

    def fake_remove(ticker):
        called['ticker'] = ticker

    monkeypatch.setattr('telegram_bot.remove_ticker', fake_remove)
    update = types.SimpleNamespace(message=DummyMessage(''))
    context = types.SimpleNamespace(args=['nvda'])
    asyncio.run(cmd_remove(update, context))
    assert called['ticker'] == 'NVDA'
    assert update.message.replies == ['NVDA removed.']


def test_cmd_list(monkeypatch):
    monkeypatch.setattr('telegram_bot.list_tickers', lambda: ['NVDA', 'AMZN'])
    update = types.SimpleNamespace(message=DummyMessage(''))
    context = types.SimpleNamespace(args=[])
    asyncio.run(cmd_list(update, context))
    assert update.message.replies == ['NVDA, AMZN']


def test_cmd_alerts(monkeypatch):
    sample = [types.SimpleNamespace(id=1, ticker='NVDA', op='>', price=100.0, active=True)]
    monkeypatch.setattr('telegram_bot.list_alerts', lambda: sample)
    update = types.SimpleNamespace(message=DummyMessage(''))
    context = types.SimpleNamespace(args=[])
    asyncio.run(cmd_alerts(update, context))
    assert update.message.replies == ['1:NVDA>100.0 on']
