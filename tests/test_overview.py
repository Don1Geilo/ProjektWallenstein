import os

os.environ.setdefault('CLIENT_ID', 'x')
os.environ.setdefault('CLIENT_SECRET', 'x')
os.environ.setdefault('TELEGRAM_BOT_TOKEN', 'x')

from wallenstein.overview import generate_overview


def test_generate_overview_starts_with_chart_emoji(monkeypatch):
    def fake_get_latest_prices(db_path, tickers, use_eur=False):
        return {t: (1.11 if use_eur else 2.22) for t in tickers}

    def fake_update_reddit_data(tickers):
        return {t: [] for t in tickers}

    monkeypatch.setattr('wallenstein.overview.get_latest_prices', fake_get_latest_prices)
    monkeypatch.setattr('wallenstein.overview.update_reddit_data', fake_update_reddit_data)

    result = generate_overview(['NVDA'])
    assert result.startswith("📊 Wallenstein Übersicht\n")


def test_generate_overview_fetches_missing_price(monkeypatch):
    def fake_get_latest_prices(db_path, tickers, use_eur=False):
        return {t: None for t in tickers}

    def fake_update_reddit_data(tickers):
        return {t: [] for t in tickers}

    monkeypatch.setattr('wallenstein.overview.get_latest_prices', fake_get_latest_prices)
    monkeypatch.setattr('wallenstein.overview.update_reddit_data', fake_update_reddit_data)
    monkeypatch.setattr('wallenstein.overview._fetch_latest_price', lambda t: 42.0)
    monkeypatch.setattr('wallenstein.overview._fetch_usd_per_eur_rate', lambda: 2.0)

    result = generate_overview(['MSFT'])
    assert 'MSFT: 42.00 USD (21.00 EUR)' in result
