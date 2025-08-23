"""Utility functions for simplistic sentiment analysis."""

from __future__ import annotations

from typing import Dict, Iterable


def analyze_sentiment(text: str) -> float:
    """Return a dummy sentiment score for ``text``.

    The function flags positive sentiment when the words ``"long"`` or
    ``"call"`` are present and negative sentiment for ``"short"`` or
    ``"put"``.
    """

    text = text.lower()
    score = 0
    if "long" in text or "call" in text:
        score += 1
    if "short" in text or "put" in text:
        score -= 1
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

