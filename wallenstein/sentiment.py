"""Utility functions for simplistic sentiment analysis."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Iterable

import pandas as pd

from .stock_keywords import global_synonyms

# keyword -> sentiment score mapping used by :func:`analyze_sentiment`
KEYWORD_SCORES: Dict[str, int] = {
    "long": 1,
    "call": 1,
    "bull": 1,
    "bullish": 1,
    "short": -1,
    "put": -1,
    "bear": -1,
    "bearish": -1,
}


def analyze_sentiment(text: str) -> float:
    """Return a simplistic sentiment score for ``text``.

    Positive sentiment is counted for occurrences of ``"long"``, ``"call"``,
    ``"bull"`` or ``"bullish"`` while ``"short"``, ``"put"``, ``"bear`` and
    ``"bearish"`` contribute negative sentiment.
    """

    text = text.lower()
    score = 0
    for keyword, value in KEYWORD_SCORES.items():
        score += text.count(keyword) * value
    return score


def aggregate_sentiment_by_ticker(
    ticker_texts: Dict[str, Iterable[dict]]
) -> Dict[str, float]:
    """Aggregate sentiment scores for each ticker.

    Parameters
    ----------
    ticker_texts:
        Mapping of ticker symbols to iterables of post dictionaries. Each
        dictionary should contain a ``"text"`` key whose value is analysed.

    Returns
    -------
    Dict[str, float]
        Average sentiment score for each ticker. If no texts are available for
        a ticker, ``0.0`` is returned.
    """

    result: Dict[str, float] = {}
    for ticker, entries in ticker_texts.items():
        entries = list(entries)
        if entries:
            total = 0.0
            for entry in entries:
                text = entry.get("text", "") if isinstance(entry, dict) else str(entry)
                total += analyze_sentiment(text)
            result[ticker] = total / len(entries)
        else:
            result[ticker] = 0.0
    return result


def derive_recommendation(score: float) -> str:
    """Derive a basic trading recommendation from ``score``."""

    if score > 0:
        return "Buy"
    if score < 0:
        return "Sell"
    return "Hold"


def build_daily_sentiment(posts: Iterable[dict], tickers: Iterable[str]) -> pd.DataFrame:
    """Return average daily sentiment for ``tickers`` based on ``posts``.

    Each post may contain ``title``, ``text`` and optionally ``comments``. The
    post timestamp can be provided via ``created_utc`` (datetime or timestamp)
    or ``date`` (timestamp).  Posts are matched to tickers using
    :data:`global_synonyms` and scored with :func:`analyze_sentiment`.
    """

    rows = []
    for post in posts:
        ts = post.get("created_utc") or post.get("date")
        try:
            if isinstance(ts, datetime):
                day = ts.date()
            else:
                day = datetime.fromtimestamp(float(ts), tz=timezone.utc).date()
        except Exception:
            continue

        title = post.get("title") or ""
        text = post.get("text") or ""
        comments = post.get("comments", [])
        if not isinstance(comments, list):
            comments = []
        full_text = f"{title} {text} " + " ".join(map(str, comments))
        lowered = full_text.lower()

        matched = [
            t
            for t in tickers
            if any(alias.lower() in lowered for alias in global_synonyms.get(t, [t]))
        ]
        if not matched:
            continue

        val = analyze_sentiment(full_text)
        for t in matched:
            rows.append({"Date": day, "Stock": t, "Sentiment": val})

    if not rows:
        return pd.DataFrame(columns=["Date", "Stock", "Sentiment"])

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    return (
        df.groupby(["Date", "Stock"], as_index=False)["Sentiment"].mean()
    )


__all__ = [
    "analyze_sentiment",
    "aggregate_sentiment_by_ticker",
    "derive_recommendation",
    "build_daily_sentiment",
]

