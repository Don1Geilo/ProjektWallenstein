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

    def fake_overview(tickers):
        called['tickers'] = tickers
        return 'OVERVIEW'

    monkeypatch.setattr('telegram_bot.generate_overview', fake_overview)
    update = types.SimpleNamespace(message=DummyMessage('!nvda'))
    context = types.SimpleNamespace()
    asyncio.run(handle_ticker(update, context))
    assert called['tickers'] == ['NVDA']
    assert update.message.replies == ['OVERVIEW']
