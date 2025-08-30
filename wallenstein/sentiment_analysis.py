from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("wallenstein")

# --- Keyword-Boosts (konfigurierbar) ---
POS_BOOST = {"long": 0.2, "call": 0.2}
NEG_BOOST = {"short": -0.2, "put": -0.2}
MAX_ABS_KEYWORD = 0.4


# --- Lazy imports / fallbacks ---
def _try_import_transformers():
    try:
        from transformers import pipeline  # type: ignore

        return pipeline
    except Exception:
        return None


def _ensure_vader():
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer  # type: ignore

        return SentimentIntensityAnalyzer
    except Exception:
        try:
            import nltk

            nltk.download("vader_lexicon", quiet=True)
            from nltk.sentiment import SentimentIntensityAnalyzer

            return SentimentIntensityAnalyzer
        except Exception as e2:  # pragma: no cover - best effort
            log.warning(f"VADER not available: {e2}")
            return None


def _detect_lang(text: str) -> str:
    try:
        from langdetect import detect  # type: ignore

        return detect(text)
    except Exception:
        return "en"


_clean_re = re.compile(r"(https?://\S+)|(@\w+)|[#*_>`]|(&amp;)|\s+")


def preprocess(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\n", " ")
    t = _clean_re.sub(" ", t)
    t = t.strip().lower()
    return t


@dataclass
class SentimentResult:
    score: float  # [-1, 1]
    method: str
    meta: dict[str, Any]


class SentimentEngine:
    def __init__(self) -> None:
        self._hf = None
        self._vader = None
        self._setup()

    def _setup(self) -> None:
        pipeline = _try_import_transformers()
        if pipeline:
            try:
                self._hf = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
                )
                log.info("Sentiment: using HF xlm-roberta pipeline")
            except Exception as e:  # pragma: no cover - best effort
                log.warning(f"HF pipeline unavailable: {e}")
        SIA = _ensure_vader()
        if SIA:
            try:
                self._vader = SIA()
                log.info("Sentiment: VADER ready (fallback)")
            except Exception as e:  # pragma: no cover - best effort
                log.warning(f"VADER init failed: {e}")

    def _keyword_boost(self, text: str) -> float:
        if not text:
            return 0.0
        boost = 0.0
        for k, v in POS_BOOST.items():
            if k in text:
                boost += v
        for k, v in NEG_BOOST.items():
            if k in text:
                boost += v
        return max(-MAX_ABS_KEYWORD, min(MAX_ABS_KEYWORD, boost))

    def analyze(self, raw_text: str) -> SentimentResult:
        txt = preprocess(raw_text)
        if not txt:
            return SentimentResult(0.0, "empty", {})

        lang = _detect_lang(txt)
        kw = self._keyword_boost(txt)

        if self._hf and lang in {"de", "fr", "es", "it", "en"}:
            try:
                out = self._hf(txt, top_k=None, truncation=True)
                if out and isinstance(out, list):
                    by = {
                        d["label"].lower(): d["score"] for d in out if "label" in d and "score" in d
                    }
                    pos = by.get("positive", 0.0)
                    neu = by.get("neutral", 0.0)
                    neg = by.get("negative", 0.0)
                    score = pos * 1.0 + neu * 0.0 + neg * (-1.0)
                    score = max(-1.0, min(1.0, score + kw))
                    return SentimentResult(
                        score, "hf-xlm-roberta", {"lang": lang, "scores": by, "kw": kw}
                    )
            except Exception as e:  # pragma: no cover - best effort
                log.debug(f"HF inference failed, fallback to VADER: {e}")

        if self._vader:
            pol = self._vader.polarity_scores(txt)
            score = pol.get("compound", 0.0)
            score = max(-1.0, min(1.0, score + kw))
            return SentimentResult(score, "vader", {"lang": lang, "polarity": pol, "kw": kw})

        return SentimentResult(0.0 + kw, "rule-only", {"lang": lang, "kw": kw})


engine_singleton: SentimentEngine | None = None


def analyze_sentiment(text: str) -> float:
    global engine_singleton
    if engine_singleton is None:
        engine_singleton = SentimentEngine()
    return engine_singleton.analyze(text).score


def post_weight(upvotes: int = 0, num_comments: int = 0) -> float:
    return 1.0 + math.log10(1 + max(0, upvotes)) + 0.2 * math.log10(1 + max(0, num_comments))


__all__ = ["analyze_sentiment", "post_weight", "SentimentEngine", "SentimentResult"]
