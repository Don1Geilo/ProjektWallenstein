"""Utility functions for simplistic sentiment analysis.

The keyword map supports both common English and German trading slang
allowing rudimentary multilingual sentiment detection."""

from __future__ import annotations

import importlib.util
import logging
import os
import re
from collections.abc import Iterable
from functools import lru_cache

from wallenstein.config import settings

logger = logging.getLogger(__name__)
_keyword_hint_logged = False

_url_re = re.compile(r"https?://\S+")


@lru_cache(maxsize=1024)
def _preprocess(text: str) -> str:
    """Return a normalised version of ``text`` suitable for analysis."""

    text = _url_re.sub(" ", text)
    return text.lower().strip()


def _transformers_available() -> bool:
    """Return ``True`` if the ``transformers`` package can be imported."""

    return importlib.util.find_spec("transformers") is not None


def _use_bert() -> bool:
    env = os.getenv("USE_BERT_SENTIMENT")
    if env is not None:
        return env.lower() in {"1", "true", "yes"}
    return settings.USE_BERT_SENTIMENT


def _log_keyword_hint(message: str = "Using keyword-based sentiment analysis") -> None:
    """Log ``message`` once to inform about fallback behaviour."""

    global _keyword_hint_logged
    if not _keyword_hint_logged:
        logger.info(message)
        _keyword_hint_logged = True


def _use_bert_sentiment() -> bool:
    """Return ``True`` if BERT sentiment analysis is requested."""

    env = os.getenv("USE_BERT_SENTIMENT")
    if env is not None:
        return env.lower() in ("1", "true", "yes")
    return settings.USE_BERT_SENTIMENT


# Intensifiers and negation markers used to enrich the keyword map
INTENSITY_WEIGHTS: dict[str, int] = {
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
KEYWORD_SCORES: dict[str, int] = {
    # positive
    "long": 1,
    "call": 1,
    "bull": 1,
    "bullish": 1,
    "buy": 1,
    "kaufen": 1,
    "kauf": 1,
    "bullisch": 1,
    "moon": 1,
    "pump": 1,
    "yolo": 1,
    "fomo": 1,
    "hodl": 1,
    "btfd": 1,
    "rocket": 1,
    "squeeze": 1,
    "lfg": 1,
    # negative
    "short": -1,
    "put": -1,
    "bear": -1,
    "bearish": -1,
    "sell": -1,
    "verkaufen": -1,
    "bärisch": -1,
    "dip": -1,
    "dump": -1,
    "dumping": -1,
    "fud": -1,
    "crash": -1,
    "bagholder": -1,
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
        backend = settings.SENTIMENT_BACKEND
        if backend == "finbert":
            self.model = "ProsusAI/finbert"
        elif backend == "de-bert":
            self.model = "oliverguhr/german-sentiment-bert"
        elif backend == "finetuned-finbert":
            from pathlib import Path

            # resolve path relative to repository root
            self.model = str(
                Path(__file__).resolve().parent.parent / "models" / "finetuned-finbert"
            )
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

    The function prefers a BERT model when available. Users can explicitly
    select the method via the ``USE_BERT_SENTIMENT`` environment variable.
    Truthy values force the BERT model, falsy values enforce the keyword
    approach.  When unspecified the function automatically uses BERT if the
    :mod:`transformers` package is installed, otherwise it falls back to the
    keyword based implementation and logs an informational hint.
    """


    if _use_bert_sentiment():

    if _use_bert():

        try:
            return analyze_sentiment_bert(text)
        except Exception:  # pragma: no cover - defensive
            _log_keyword_hint("BERT sentiment requested but unavailable; using keyword approach")
    else:
        if _transformers_available():
            try:
                return analyze_sentiment_bert(text)
            except Exception:  # pragma: no cover - defensive
                _log_keyword_hint("BERT sentiment failed to load; using keyword approach")
        else:
            _log_keyword_hint("transformers not installed; using keyword sentiment analysis")

    return _keyword_score_cached(_preprocess(text))


def _keyword_score(text: str) -> float:
    text = apply_negation(text)
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


_keyword_score_cached = lru_cache(maxsize=2048)(_keyword_score)


def analyze_sentiment_batch(texts: list[str]) -> list[float]:
    """Return sentiment scores for multiple texts.

    When a BERT backend is available the underlying pipeline processes the
    whole ``texts`` list in a single call to avoid repeated model invocations.
    If keyword based sentiment analysis is active, the function simply maps
    :func:`analyze_sentiment` over the inputs.
    """

    global _bert_analyzer

    if _use_bert_sentiment():

    if _use_bert():

        try:
            if _bert_analyzer is None:
                _bert_analyzer = BertSentiment()
            results = _bert_analyzer(texts)
        except Exception:  # pragma: no cover - defensive
            _log_keyword_hint("BERT sentiment requested but unavailable; using keyword approach")
            return [_keyword_score_cached(_preprocess(t)) for t in texts]
    else:
        if _transformers_available():
            try:
                if _bert_analyzer is None:
                    _bert_analyzer = BertSentiment()
                results = _bert_analyzer(texts)
            except Exception:  # pragma: no cover - defensive
                _log_keyword_hint("BERT sentiment failed to load; using keyword approach")
                return [_keyword_score_cached(_preprocess(t)) for t in texts]
        else:
            _log_keyword_hint("transformers not installed; using keyword sentiment analysis")
            return [_keyword_score_cached(_preprocess(t)) for t in texts]

    scores: list[float] = []
    for data in results:
        label = str(data.get("label", "")).lower()
        score = float(data.get("score", 0.0))
        if "positive" in label:
            scores.append(score)
        elif "negative" in label:
            scores.append(-score)
        else:
            scores.append(0.0)
    return scores


def aggregate_sentiment_by_ticker(ticker_texts: dict[str, Iterable[dict]]) -> dict[str, float]:
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

    result: dict[str, float] = {}
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
    "analyze_sentiment_batch",
    "analyze_sentiment_bert",
    "aggregate_sentiment_by_ticker",
    "derive_recommendation",
    "BertSentiment",
]
