"""Generate price and sentiment overview text for tickers."""
from __future__ import annotations

import os
from typing import List

from .db_utils import get_latest_prices
from .reddit_scraper import update_reddit_data
from .sentiment import analyze_sentiment_batch, derive_recommendation

DB_PATH = os.getenv("WALLENSTEIN_DB_PATH", "data/wallenstein.duckdb").strip()


def generate_overview(tickers: List[str]) -> str:
    """Return a formatted overview for ``tickers``.

    The overview lists the latest USD prices and a simple Reddit-based
    sentiment score with a derived recommendation for each ticker.
    """

    prices_usd = get_latest_prices(DB_PATH, tickers, use_eur=False)
    try:
        reddit_posts = update_reddit_data(tickers)
    except Exception:  # pragma: no cover - network or config issues
        reddit_posts = {t: [] for t in tickers}

    sentiments = {}
    for ticker in tickers:
        posts = reddit_posts.get(ticker, [])
        texts = [p.get("text", "") for p in posts if p.get("text")]
        if texts:
            scores = analyze_sentiment_batch(texts)
            sentiments[ticker] = sum(scores) / len(scores)
        else:
            sentiments[ticker] = 0.0

    price_lines = []
    sentiment_lines = []
    for t in tickers:
        price = prices_usd.get(t)
        if price is not None:
            price_lines.append(f"{t}: {price:.2f} USD")
        else:
            price_lines.append(f"{t}: n/a")

        sent = sentiments.get(t, 0.0)
        rec = derive_recommendation(sent)
        sentiment_lines.append(f"{t}: Sentiment {sent:+.2f} | {rec}")

    return "\ud83d\udcca Wallenstein Übersicht\n" + "\n".join(price_lines + sentiment_lines)
