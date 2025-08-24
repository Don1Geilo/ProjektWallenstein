import asyncio
import types

from telegram_bot import handle_ticker


class DummyMessage:
    def __init__(self, text: str):
        self.text = text
        self.replies = []

    async def reply_text(self, text: str) -> None:
        self.replies.append(text)


def test_handle_ticker(monkeypatch):
    called = {}

    def fake_update(tickers):
        called['tickers'] = tickers
        return {tickers[0]: [1, 2]}

    monkeypatch.setattr('telegram_bot.update_reddit_data', fake_update)
    monkeypatch.setattr('telegram_bot.notify_telegram', lambda msg: None)
    update = types.SimpleNamespace(message=DummyMessage('!nvda'))
    context = types.SimpleNamespace()
    asyncio.run(handle_ticker(update, context))
    assert called['tickers'] == ['NVDA']
    assert update.message.replies == ['NVDA: 2 Reddit posts gefunden.']
