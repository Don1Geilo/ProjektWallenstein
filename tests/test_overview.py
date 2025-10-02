import os

import duckdb

os.environ.setdefault('CLIENT_ID', 'x')
os.environ.setdefault('CLIENT_SECRET', 'x')
os.environ.setdefault('REDDIT_CLIENT_ID', 'x')
os.environ.setdefault('REDDIT_CLIENT_SECRET', 'x')
os.environ.setdefault('REDDIT_USER_AGENT', 'x')
os.environ.setdefault('TELEGRAM_BOT_TOKEN', 'x')

from wallenstein.overview import generate_overview


def test_generate_overview_starts_with_chart_emoji(monkeypatch):
    def fake_get_latest_prices(db_path, tickers, use_eur=False):
        return {t: (1.11 if use_eur else 2.22) for t in tickers}

    monkeypatch.setattr('wallenstein.overview.get_latest_prices', fake_get_latest_prices)
    monkeypatch.setattr(
        'wallenstein.overview.fetch_weekly_returns',
        lambda *args, **kwargs: {str(sym).upper(): 0.05 for sym in args[1]},
    )

    result = generate_overview(['NVDA'], reddit_posts={'NVDA': []})
    assert result.startswith("📊 Wallenstein Markt-Update\n")


def test_generate_overview_fetches_missing_price(monkeypatch):
    def fake_get_latest_prices(db_path, tickers, use_eur=False):
        return {t: None for t in tickers}

    monkeypatch.setattr('wallenstein.overview.get_latest_prices', fake_get_latest_prices)
    monkeypatch.setattr('wallenstein.overview._fetch_latest_price', lambda t: 42.0)
    monkeypatch.setattr('wallenstein.overview._fetch_usd_per_eur_rate', lambda: 2.0)
    monkeypatch.setattr(
        'wallenstein.overview.fetch_weekly_returns',
        lambda *args, **kwargs: {str(sym).upper(): 0.05 for sym in args[1]},
    )

    result = generate_overview(['MSFT'], reddit_posts={'MSFT': []})
    assert 'Preis: 42.00 USD | 21.00 EUR' in result


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
    monkeypatch.setattr(
        'wallenstein.overview.fetch_weekly_returns',
        lambda *args, **kwargs: {str(sym).upper(): 0.05 for sym in args[1]},
    )

    result = generate_overview(['NVDA'], reddit_posts={'NVDA': []})
    assert 'Sentiment 1d: +0.50' in result


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
    monkeypatch.setattr(
        'wallenstein.overview.fetch_weekly_returns',
        lambda *args, **kwargs: {str(sym).upper(): 0.05 for sym in args[1]},
    )

    result = generate_overview(['NVDA'], reddit_posts={'NVDA': []})
    assert 'Alias: nvidia' in result


def test_generate_overview_includes_trending_section(monkeypatch, tmp_path):
    db_path = tmp_path / 'db.duckdb'
    con = duckdb.connect(str(db_path))
    con.execute(
        "CREATE TABLE reddit_trends (date DATE, ticker VARCHAR, mentions INTEGER, avg_upvotes DOUBLE, hotness DOUBLE)"
    )
    con.execute(
        "INSERT INTO reddit_trends VALUES (CURRENT_DATE, 'TSLA', 5, 10.0, 100.0)"
    )
    con.close()

    monkeypatch.setattr(
        'wallenstein.overview.get_latest_prices',
        lambda db_path, tickers, use_eur=False: {t: 1.0 for t in tickers},
    )
    monkeypatch.setattr('wallenstein.overview.DB_PATH', str(db_path))
    monkeypatch.setattr(
        'wallenstein.overview.fetch_weekly_returns',
        lambda *args, **kwargs: {str(sym).upper(): 0.05 for sym in args[1]},
    )

    result = generate_overview(['TSLA'], reddit_posts={'TSLA': []})
    assert '🔥 Reddit Trends:' in result
    assert '- TSLA: 5 Mentions' in result
    assert '7d +5.0%' in result


def test_generate_overview_lists_multi_hits(monkeypatch, tmp_path):
    db_path = tmp_path / 'db.duckdb'
    duckdb.connect(str(db_path)).close()

    monkeypatch.setattr(
        'wallenstein.overview.get_latest_prices',
        lambda db_path, tickers, use_eur=False: {t: 1.0 for t in tickers},
    )
    monkeypatch.setattr('wallenstein.overview.DB_PATH', str(db_path))
    captured: dict[str, list[str]] = {}

    def fake_fetch_weekly_returns(con, symbols, max_symbols=None):
        seq = [str(sym) for sym in symbols]
        captured['symbols'] = seq
        return {str(sym).upper(): 0.05 for sym in seq}

    monkeypatch.setattr(
        'wallenstein.overview.fetch_weekly_returns', fake_fetch_weekly_returns,
    )

    result = generate_overview(
        ['MSFT'], reddit_posts={'NVDA': [{}, {}]}
    )
    assert '🔁 Mehrfach erwähnt:' in result
    assert '- NVDA: 2 Posts, 7d +5.0%' in result
    assert 'NVDA' in captured['symbols']

    assert '- NVDA: 2 Posts' in result



def test_generate_overview_includes_weekly_line(monkeypatch, tmp_path):
    db_path = tmp_path / 'db.duckdb'
    duckdb.connect(str(db_path)).close()

    monkeypatch.setattr(
        'wallenstein.overview.get_latest_prices',
        lambda db_path, tickers, use_eur=False: {t: 10.0 for t in tickers},
    )
    monkeypatch.setattr('wallenstein.overview.DB_PATH', str(db_path))
    monkeypatch.setattr(
        'wallenstein.overview.fetch_weekly_returns',
        lambda *args, **kwargs: {str(sym).upper(): 0.12 for sym in args[1]},
    )

    result = generate_overview(['AMZN'], reddit_posts={'AMZN': []})
    assert 'Trend 7d: +12.0%' in result


def test_generate_overview_includes_ml_predictions(monkeypatch, tmp_path):
    db_path = tmp_path / 'db.duckdb'
    con = duckdb.connect(str(db_path))
    con.execute(
        "CREATE TABLE predictions (as_of TIMESTAMP, ticker TEXT, horizon_days INT, signal TEXT, confidence DOUBLE, expected_return DOUBLE, version TEXT)"
    )
    con.execute(
        "CREATE TABLE model_training_state (ticker TEXT, latest_price_date DATE, price_row_count INT, latest_sentiment_date DATE, sentiment_row_count INT, latest_post_utc TIMESTAMP, trained_at TIMESTAMP, accuracy DOUBLE, f1 DOUBLE, roc_auc DOUBLE, precision_score DOUBLE, recall_score DOUBLE, avg_strategy_return DOUBLE, long_win_rate DOUBLE)"
    )
    con.execute(
        "INSERT INTO predictions VALUES (TIMESTAMP '2024-03-01 12:00:00', 'NVDA', 1, 'buy', 0.72, 0.015, 'ml-v2')"
    )
    con.execute(
        "INSERT INTO model_training_state VALUES ('NVDA', CURRENT_DATE, 100, CURRENT_DATE, 80, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 0.68, 0.62, 0.70, 0.64, 0.59, 0.02, 0.6)"
    )
    con.close()

    monkeypatch.setattr(
        'wallenstein.overview.get_latest_prices',
        lambda db_path, tickers, use_eur=False: {t: 10.0 for t in tickers},
    )
    monkeypatch.setattr('wallenstein.overview.DB_PATH', str(db_path))
    monkeypatch.setattr(
        'wallenstein.overview.fetch_weekly_returns',
        lambda *args, **kwargs: {str(sym).upper(): 0.05 for sym in args[1]},
    )

    result = generate_overview(['NVDA'], reddit_posts={'NVDA': []})
    assert '🚦 ML Signale (1d Horizont):' in result
    assert '- NVDA: 72.0% Conviction' in result
    assert 'Erwartung +1.50%' in result
    assert 'Backtest Ø +2.00%' in result
    assert 'Trefferquote 60.0%' in result
    assert 'ml-v2' in result
