import importlib
import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_hourly_trend_preview_includes_weekly(tmp_path, monkeypatch):
    os.environ.setdefault("REDDIT_CLIENT_ID", "x")
    os.environ.setdefault("REDDIT_CLIENT_SECRET", "x")
    os.environ.setdefault("REDDIT_USER_AGENT", "x")

    if "main" in sys.modules:
        main = importlib.reload(sys.modules["main"])
    else:
        main = importlib.import_module("main")

    db_path = tmp_path / "telegram_trends.duckdb"
    monkeypatch.setattr(main, "DB_PATH", str(db_path), raising=False)
    main.init_schema(str(db_path))

    monkeypatch.setattr(main, "enrich_reddit_posts", lambda *args, **kwargs: 0)
    monkeypatch.setattr(main, "compute_reddit_trends", lambda *args, **kwargs: 0)
    monkeypatch.setattr(main, "compute_returns", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        main, "compute_reddit_sentiment", lambda *args, **kwargs: (0, 0)
    )

    messages: list[str] = []

    def fake_notify(message: str) -> bool:
        messages.append(message)
        return True

    monkeypatch.setattr(main, "notify_telegram", fake_notify)

    candidate = types.SimpleNamespace(
        symbol="AAPL",
        mentions_24h=42,
        lift=3.4,
        trend=9.1,
        is_known=True,
        weekly_return=None,
    )

    monkeypatch.setattr(
        main,
        "scan_reddit_for_candidates",
        lambda *args, **kwargs: [candidate],
    )
    monkeypatch.setattr(
        main,
        "auto_add_candidates_to_watchlist",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        main,
        "fetch_weekly_returns",
        lambda *args, **kwargs: {"AAPL": 0.05},
    )

    main.generate_trends({"AAPL": [{"id": "p1"}]})

    assert messages, "hourly notification was not emitted"
    assert "7d +5.0%" in messages[0]


def test_auto_sentiment_formatter(monkeypatch):
    os.environ.setdefault("REDDIT_CLIENT_ID", "x")
    os.environ.setdefault("REDDIT_CLIENT_SECRET", "x")
    os.environ.setdefault("REDDIT_USER_AGENT", "x")

    if "main" in sys.modules:
        main = importlib.reload(sys.modules["main"])
    else:
        main = importlib.import_module("main")

    posts = [
        {"sentiment": 0.5},
        {"sentiment": -0.1},
        {"sentiment": 0.1},
        {"sentiment": "0.2"},
    ]
    candidate = types.SimpleNamespace(symbol="ABC", mentions_24h=42, lift=5.2)

    line = main._format_auto_sentiment_line("ABC", posts, candidate)

    assert "Ø +0.17" in line
    assert "Median +0.15" in line
    assert "4 Beiträge" in line
    assert "m24h=42" in line
    assert "Lift x5.2" in line


def test_auto_sentiment_formatter_without_scores(monkeypatch):
    os.environ.setdefault("REDDIT_CLIENT_ID", "x")
    os.environ.setdefault("REDDIT_CLIENT_SECRET", "x")
    os.environ.setdefault("REDDIT_USER_AGENT", "x")

    if "main" in sys.modules:
        main = importlib.reload(sys.modules["main"])
    else:
        main = importlib.import_module("main")

    posts = [{"sentiment": None}, {}, {"sentiment": "nan"}]

    line = main._format_auto_sentiment_line("DEF", posts, None)

    assert "keine Sentiment-Daten" in line
    assert "(3 Beiträge)" in line
