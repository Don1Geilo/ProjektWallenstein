"""Utility functions for simplistic sentiment analysis."""

from __future__ import annotations

from typing import Dict, Iterable

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


def aggregate_sentiment_by_ticker(ticker_texts: Dict[str, Iterable[str]]) -> Dict[str, float]:
    """Aggregate sentiment scores for each ticker.

    Parameters
    ----------
    ticker_texts:
        Mapping of ticker symbols to iterables of text snippets.

    Returns
    -------
    Dict[str, float]
        Average sentiment score for each ticker. If no texts are available for
        a ticker, ``0.0`` is returned.
    """

    result: Dict[str, float] = {}
    for ticker, texts in ticker_texts.items():
        texts = list(texts)
        if texts:
            result[ticker] = sum(analyze_sentiment(t) for t in texts) / len(texts)
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


__all__ = [
    "analyze_sentiment",
    "aggregate_sentiment_by_ticker",
    "derive_recommendation",
]

