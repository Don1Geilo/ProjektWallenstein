from __future__ import annotations

import logging
import math
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("wallenstein")

# --- Keyword-Boosts (konfigurierbar) ---
POS_BOOST = {"long": 0.2, "call": 0.2}
NEG_BOOST = {"short": -0.2, "put": -0.2}
MAX_ABS_KEYWORD = 0.4

# Für stabilere Tokenizer-Läufe in CI
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# --------------------------
# Lazy imports / fallbacks
# --------------------------
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
            import nltk  # type: ignore

            nltk.download("vader_lexicon", quiet=True)
            from nltk.sentiment import SentimentIntensityAnalyzer  # type: ignore

            return SentimentIntensityAnalyzer
        except Exception as e2:  # pragma: no cover
            log.warning(f"VADER not available: {e2}")
            return None


def _detect_lang(text: str) -> str:
    try:
        from langdetect import detect  # type: ignore

        return detect(text)
    except Exception:
        # Fallback: Englisch annehmen
        return "en"


_clean_re = re.compile(r"(https?://\S+)|(@\w+)|[#*`>|]|(&amp;)|\s+")


# Cashtags ($NVDA) und Zahlen/Buchstaben bleiben erhalten.
def preprocess(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\n", " ")
    t = _clean_re.sub(" ", t)
    t = t.strip().lower()
    return t


# Wortgrenzen-Regexes für Keyword-Boosts
def _compile_kw_patterns(d: dict[str, float]) -> list[tuple[re.Pattern, float]]:
    pats: list[tuple[re.Pattern, float]] = []
    for k, v in d.items():
        # \b passt auf Wortgrenzen; erlaubt casings bereits durch lowercase in preprocess
        pats.append((re.compile(rf"\b{re.escape(k)}\b"), v))
    return pats


_POS_PATTERNS = _compile_kw_patterns(POS_BOOST)
_NEG_PATTERNS = _compile_kw_patterns(NEG_BOOST)


def _keyword_boost(text: str) -> float:
    if not text:
        return 0.0
    boost = 0.0
    for pat, val in _POS_PATTERNS:
        if pat.search(text):
            boost += val
    for pat, val in _NEG_PATTERNS:
        if pat.search(text):
            boost += val
    # clamp
    return max(-MAX_ABS_KEYWORD, min(MAX_ABS_KEYWORD, boost))


@dataclass
class SentimentResult:
    score: float  # [-1, 1]
    method: str
    meta: dict[str, Any]


class SentimentEngine:
    """
    Routing:
      - EN: ProsusAI/finbert (finance-tuned, kein SentencePiece-Zwang)
      - DE/FR/ES/IT/EN: CardiffNLP XLM-R (multilingual)
      - Fallback: VADER
    Modelle werden lazy geladen (erst beim ersten Bedarf).
    """

    FINBERT_ID = "ProsusAI/finbert"
    XLMR_ID = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

    def __init__(self) -> None:
        self._pipeline = _try_import_transformers()
        self._hf_finbert: Callable | None = None
        self._hf_xlmr: Callable | None = None
        self._vader = None
        self._setup_vader()

    def _setup_vader(self) -> None:
        SIA = _ensure_vader()
        if SIA:
            try:
                self._vader = SIA()
                log.info("Sentiment: VADER ready (fallback)")
            except Exception as e:  # pragma: no cover
                log.warning(f"VADER init failed: {e}")

    # Lazy Loader
    def _get_finbert(self):
        if self._hf_finbert or not self._pipeline:
            return self._hf_finbert
        try:
            self._hf_finbert = self._pipeline(
                "sentiment-analysis",
                model=self.FINBERT_ID,
                top_k=None,
                truncation=True,
                function_to_apply="softmax",
            )
            log.info("Sentiment: using HF pipeline %s", self.FINBERT_ID)
        except Exception as e:
            log.warning(f"HF FinBERT unavailable: {e}")
            self._hf_finbert = None
        return self._hf_finbert

    def _get_xlmr(self):
        if self._hf_xlmr or not self._pipeline:
            return self._hf_xlmr
        try:
            self._hf_xlmr = self._pipeline(
                "sentiment-analysis",
                model=self.XLMR_ID,
                top_k=None,
                truncation=True,
                function_to_apply="softmax",
            )
            log.info("Sentiment: using HF pipeline %s", self.XLMR_ID)
        except Exception as e:
            log.warning(f"HF XLM-R pipeline unavailable: {e}")
            self._hf_xlmr = None
        return self._hf_xlmr

    @staticmethod
    def _scores_to_scalar(scores: list[dict[str, Any]]) -> tuple[float, dict[str, float]]:
        # erwartet Liste von {"label": "...", "score": float}
        by = {}
        for d in scores:
            if isinstance(d, dict) and "label" in d and "score" in d:
                by[str(d["label"]).lower()] = float(d["score"])
        # übliche Labels: positive/neutral/negative
        pos = by.get("positive", 0.0)
        neg = by.get("negative", 0.0)
        # neutral fließt nicht in den Skalar ein (0-Gewicht)
        return (max(-1.0, min(1.0, pos - neg))), by

    def analyze(self, raw_text: str) -> SentimentResult:
        txt = preprocess(raw_text)
        if not txt:
            return SentimentResult(0.0, "empty", {})

        lang = _detect_lang(txt)
        kw = _keyword_boost(txt)

        # 1) HF bevorzugt (sprachabhängiges Routing)
        if self._pipeline:
            try:
                if lang == "en":
                    pipe = self._get_finbert() or self._get_xlmr()
                    model_id = self.FINBERT_ID if self._hf_finbert else self.XLMR_ID
                else:
                    pipe = self._get_xlmr() or self._get_finbert()
                    model_id = self.XLMR_ID if self._hf_xlmr else self.FINBERT_ID

                if pipe:
                    out = pipe(txt)
                    # pipeline gibt List[List[dict]] oder List[dict]; normal: List[dict]
                    if isinstance(out, list) and out and isinstance(out[0], (list, tuple)):
                        scores = out[0]
                    else:
                        scores = out
                    scalar, by = self._scores_to_scalar(scores)  # type: ignore[arg-type]
                    score = max(-1.0, min(1.0, scalar + kw))
                    return SentimentResult(
                        score, f"hf:{model_id}", {"lang": lang, "scores": by, "kw": kw}
                    )
            except Exception as e:  # pragma: no cover
                log.debug(f"HF inference failed, fallback to VADER: {e}")

        # 2) VADER Fallback
        if self._vader:
            pol = self._vader.polarity_scores(txt)
            score = pol.get("compound", 0.0)
            score = max(-1.0, min(1.0, score + kw))
            return SentimentResult(score, "vader", {"lang": lang, "polarity": pol, "kw": kw})

        # 3) Rule-only
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
