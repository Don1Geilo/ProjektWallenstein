"""Utility functions for simplistic sentiment analysis.

The keyword map supports both common English and German trading slang
allowing rudimentary multilingual sentiment detection."""

from __future__ import annotations

from typing import Dict, Iterable

# keyword -> sentiment score mapping used by :func:`analyze_sentiment`
#
# Many retail traders express their sentiment using simple words like
# ``"buy"`` or ``"sell"``.  The previous implementation only looked for
# option‑related terms (``"call"``/``"put"``) or metaphors such as
# ``"bullish"`` and therefore returned ``0`` for common phrases like
# "buy the dip".  This resulted in neutral recommendations even when the
# text clearly contained a directional bias.  To improve coverage we map
# a couple of additional keywords to sentiment scores.
KEYWORD_SCORES: Dict[str, int] = {
    # positive
    "long": 1,
    "call": 1,
    "bull": 1,
    "bullish": 1,
    "buy": 1,
    "kaufen": 1,
    "bullisch": 1,
    # negative
    "short": -1,
    "put": -1,
    "bear": -1,
    "bearish": -1,
    "sell": -1,
    "verkaufen": -1,
    "bärisch": -1,
}


def analyze_sentiment(text: str) -> float:
    """Return a simplistic sentiment score for ``text``.

    Positive sentiment is counted for occurrences of words such as ``"long"``,
    ``"call"``, ``"bullish"`` or ``"buy"``.  Negative sentiment is triggered by
    phrases like ``"short"``, ``"put"``, ``"bearish"`` or ``"sell"``.
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


__all__ = [
    "analyze_sentiment",
    "aggregate_sentiment_by_ticker",
    "derive_recommendation",
]

