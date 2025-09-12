import os

import duckdb

os.environ.setdefault('CLIENT_ID', 'x')
os.environ.setdefault('CLIENT_SECRET', 'x')
os.environ.setdefault('TELEGRAM_BOT_TOKEN', 'x')

from wallenstein.overview import generate_overview


def test_generate_overview_starts_with_chart_emoji(monkeypatch):
    def fake_get_latest_prices(db_path, tickers, use_eur=False):
        return {t: (1.11 if use_eur else 2.22) for t in tickers}

    monkeypatch.setattr('wallenstein.overview.get_latest_prices', fake_get_latest_prices)

    result = generate_overview(['NVDA'], reddit_posts={'NVDA': []})
    assert result.startswith("📊 Wallenstein Übersicht\n")


def test_generate_overview_fetches_missing_price(monkeypatch):
    def fake_get_latest_prices(db_path, tickers, use_eur=False):
        return {t: None for t in tickers}

    monkeypatch.setattr('wallenstein.overview.get_latest_prices', fake_get_latest_prices)
    monkeypatch.setattr('wallenstein.overview._fetch_latest_price', lambda t: 42.0)
    monkeypatch.setattr('wallenstein.overview._fetch_usd_per_eur_rate', lambda: 2.0)

    result = generate_overview(['MSFT'], reddit_posts={'MSFT': []})
    assert 'MSFT: 42.00 USD (21.00 EUR)' in result


def test_generate_overview_includes_latest_sentiment(monkeypatch, tmp_path):
    db_path = tmp_path / 'db.duckdb'
    con = duckdb.connect(str(db_path))
    con.execute(
        "CREATE TABLE reddit_sentiment_daily (ticker VARCHAR, date DATE, sentiment_weighted DOUBLE)"
    )
    con.execute(
        "INSERT INTO reddit_sentiment_daily VALUES ('NVDA', CURRENT_DATE, 0.5)"
    )
    con.close()

    monkeypatch.setattr(
        'wallenstein.overview.get_latest_prices',
        lambda db_path, tickers, use_eur=False: {t: 1.0 for t in tickers},
    )
    monkeypatch.setattr('wallenstein.overview.DB_PATH', str(db_path))

    result = generate_overview(['NVDA'], reddit_posts={'NVDA': []})
    assert 'Sentiment (1d, weighted): +0.50' in result


def test_generate_overview_lists_aliases(monkeypatch, tmp_path):
    db_path = tmp_path / 'db.duckdb'
    con = duckdb.connect(str(db_path))
    con.execute(
        "CREATE TABLE ticker_aliases (ticker VARCHAR, alias VARCHAR, source VARCHAR)"
    )
    con.execute(
        "INSERT INTO ticker_aliases VALUES ('NVDA', 'nvidia', 'seed')"
    )
    con.close()

    monkeypatch.setattr(
        'wallenstein.overview.get_latest_prices',
        lambda db_path, tickers, use_eur=False: {t: 1.0 for t in tickers},
    )
    monkeypatch.setattr('wallenstein.overview.DB_PATH', str(db_path))

    result = generate_overview(['NVDA'], reddit_posts={'NVDA': []})
    assert 'Alias: nvidia' in result
