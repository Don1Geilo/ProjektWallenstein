"""Utility functions for simplistic sentiment analysis.

The keyword map supports both common English and German trading slang
allowing rudimentary multilingual sentiment detection."""

from __future__ import annotations

import os
from typing import Dict, Iterable

# Intensifiers and negation markers used to enrich the keyword map
INTENSITY_WEIGHTS: Dict[str, int] = {
    "strong": 2,
    "massiv": 2,
}

NEGATION_MARKERS = {"nicht", "kein", "no", "not"}

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
    "kauf": 1,
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

# extend keyword scores with simple intensity phrases and negated forms
for _word, _score in list(KEYWORD_SCORES.items()):
    for _intens, _mult in INTENSITY_WEIGHTS.items():
        KEYWORD_SCORES[f"{_intens} {_word}"] = _score * _mult
    KEYWORD_SCORES[f"not_{_word}"] = -_score


def apply_negation(text: str) -> str:
    """Collapse negations so that they can be scored correctly.

    The function searches for occurrences of words like ``"nicht"`` or
    ``"kein"`` and prefixes a sentiment keyword within the next two tokens
    with ``"not_"``.  Intermediate filler words are preserved while the
    negation marker itself is dropped.

    Examples
    --------
    ``"nicht kaufen"`` -> ``"not_kaufen"``
    ``"kein sell"`` -> ``"not_sell"``
    ``"nicht so bullish"`` -> ``"so not_bullish"``
    """

    words = text.split()
    result = []
    i = 0
    while i < len(words):
        word = words[i]
        if word in NEGATION_MARKERS:
            target = None
            for offset in (1, 2):
                j = i + offset
                if j < len(words):
                    candidate = words[j]
                    if candidate in KEYWORD_SCORES or candidate in INTENSITY_WEIGHTS:
                        target = j
                        break
            if target is not None:
                result.extend(words[i + 1 : target])
                result.append(f"not_{words[target]}")
                i = target + 1
                continue
            i += 1
            continue
        result.append(word)
        i += 1
    return " ".join(result)


class BertSentiment:
    """Sentiment analyzer backed by HuggingFace models.

    The underlying pipeline is instantiated lazily on first use. The model is
    chosen via the ``SENTIMENT_BACKEND`` environment variable which accepts
    ``"finbert"`` (default) or ``"de-bert"`` for a German model.
    """

    _pipe = None

    def __init__(self) -> None:
        backend = os.getenv("SENTIMENT_BACKEND", "finbert").lower()
        if backend == "finbert":
            self.model = "ProsusAI/finbert"
        elif backend == "de-bert":
            self.model = "oliverguhr/german-sentiment-bert"
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        self.backend = backend

    @property
    def pipe(self):  # pragma: no cover - heavy model
        if self._pipe is None:
            from transformers import pipeline

            self._pipe = pipeline("sentiment-analysis", model=self.model)
        return self._pipe

    def __call__(self, text: str):  # pragma: no cover - heavy model
        return self.pipe(text)


_bert_analyzer: BertSentiment | None = None


def analyze_sentiment_bert(text: str) -> float:
    """Analyse sentiment using a BERT based model."""

    global _bert_analyzer
    if _bert_analyzer is None:
        _bert_analyzer = BertSentiment()
    result = _bert_analyzer(text)
    if not result:
        return 0.0
    data = result[0]
    label = str(data.get("label", "")).lower()
    score = float(data.get("score", 0.0))
    if "positive" in label:
        return score
    if "negative" in label:
        return -score
    return 0.0


def analyze_sentiment(text: str) -> float:
    """Return a sentiment score for ``text``.

    The function defaults to a lightweight keyword approach but can switch to a
    BERT based model when the ``USE_BERT_SENTIMENT`` environment variable is
    set to a truthy value.
    """

    if os.getenv("USE_BERT_SENTIMENT", "").lower() in {"1", "true", "yes"}:
        return analyze_sentiment_bert(text)

    text = apply_negation(text.lower())
    tokens = text.split()
    score = 0
    i = 0
    while i < len(tokens):
        token = tokens[i]
        multiplier = 1
        if token in INTENSITY_WEIGHTS and i + 1 < len(tokens):
            multiplier = INTENSITY_WEIGHTS[token]
            i += 1
            token = tokens[i]
        score += KEYWORD_SCORES.get(token, 0) * multiplier
        i += 1
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
    "analyze_sentiment_bert",
    "aggregate_sentiment_by_ticker",
    "derive_recommendation",
    "BertSentiment",
]

