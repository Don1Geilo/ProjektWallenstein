# wallenstein/sentiment.py
"""
Pragmatische Sentiment-Utilities (FinBERT → VADER → Keywords).

- Bevorzugt ProsusAI/finbert (lokal in models/finbert, sonst HF-Cache)
- Robuster Fallback: VADER (mit kleinem Finanz/WSB-Boost)
- Letzter Fallback: einfache Keyword-Heuristik (EN/DE + Negation/Intensifier)
- Batch-API und tägliche Aggregation (Upvotes/Kommentare-gewichtung)
"""

from __future__ import annotations

import importlib.util
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional
from zoneinfo import ZoneInfo

# --- Optional dependencies (defensiv) ---
try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

try:
    import spacy  # type: ignore
except Exception:  # pragma: no cover
    spacy = None  # type: ignore

try:
    from nltk.stem import SnowballStemmer  # type: ignore
except Exception:  # pragma: no cover
    SnowballStemmer = None  # type: ignore

from .config import settings, ensure_hf_env

logger = logging.getLogger(__name__)

# Cached BERT analyzer instance (lazily created)
_bert_analyzer = None

# --- regex & globals ---
_url_re = re.compile(r"https?://\S+")
_spacy_nlp = None
_stemmer_de = None
_keyword_hint_logged = False

# HF-ENV setzen (Token/Cache), kein lautes Login nötig
ensure_hf_env()

# ---------------------------------------------------------------------------
# Normalisierung / Heuristiken
# ---------------------------------------------------------------------------
@lru_cache(maxsize=2048)
def normalize_token(token: str) -> str:
    if not token:
        return token
    try:
        if token in KEYWORD_SCORES or token in NEGATION_MARKERS:
            return token
    except Exception:
        pass

    # spaCy Lemmatizer (optional)
    global _spacy_nlp
    if spacy is not None and _spacy_nlp is None:
        for model in ("de_core_news_sm", "en_core_web_sm"):
            try:
                _spacy_nlp = spacy.load(model, disable=["parser", "ner", "textcat"])  # type: ignore[arg-type]
                break
            except Exception:
                continue
    if _spacy_nlp is not None:
        try:
            doc = _spacy_nlp(token)
            if doc and doc[0].lemma_:
                lemma = doc[0].lemma_.lower()
                if lemma:
                    return lemma
        except Exception:
            pass

    # Deutscher Stemmer als leichter Fallback
    global _stemmer_de
    if SnowballStemmer is not None and _stemmer_de is None:
        try:
            _stemmer_de = SnowballStemmer("german")  # type: ignore[call-arg]
        except Exception:
            _stemmer_de = False
    if _stemmer_de not in (None, False):
        try:
            token = _stemmer_de.stem(token)  # type: ignore[union-attr]
        except Exception:
            pass

    # einfache deutsch-lastige Nachkorrektur
    if len(token) > 4:
        if token.endswith("st"):
            return token[:-2] + "en"
        if token.endswith("t"):
            return token[:-1] + "en"
        if token.endswith("e"):
            return token + "n"
    return token


@lru_cache(maxsize=1024)
def _preprocess(text: str) -> str:
    """
    Lowercase, URLs raus, Interpunktion -> Space.
    Erhält Cashtags/Hashtags ($, #) und 🚀.
    """
    text = (text or "").lower()
    text = _url_re.sub(" ", text)
    # alles außer Wortzeichen, $ # und 🚀 zu Space machen
    text = re.sub(r"[^\w$#🚀]+", " ", text, flags=re.U)
    tokens = [normalize_token(tok) for tok in text.split()]
    return " ".join(tokens)


def _transformers_available() -> bool:
    return importlib.util.find_spec("transformers") is not None


def _log_keyword_hint(msg: str) -> None:
    global _keyword_hint_logged
    if not _keyword_hint_logged:
        logger.info(msg)
        _keyword_hint_logged = True


# Intensifier & Negationen (EN/DE)
INTENSITY_WEIGHTS: dict[str, int] = {
    "strong": 2, "massiv": 2, "extrem": 2, "mega": 2, "super": 2, "über": 2,
}
NEGATION_MARKERS = {"nicht", "kein", "no", "not", "nie", "ohne"}

def _load_sentiment_config(path: str | Path | None = None) -> None:
    if not yaml:
        return
    root = Path(__file__).resolve().parents[1]
    cfg = Path(path) if path else root / "data" / "sentiment_config.yaml"
    if not cfg.exists():
        return
    try:
        with cfg.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except Exception as exc:
        logger.warning("Could not load sentiment config from %s: %s", cfg, exc)
        return
    for w, k in (data.get("intensity_weights") or {}).items():
        try:
            INTENSITY_WEIGHTS[str(w).lower()] = int(k)
        except Exception:
            continue
    for m in data.get("negation_markers") or []:
        try:
            NEGATION_MARKERS.add(str(m).lower())
        except Exception:
            continue

# keyword -> sentiment score (rudimentär EN/DE trading slang)
KEYWORD_SCORES: dict[str, int] = {
    # positive
    "long": 1, "call": 1, "bull": 1, "bullish": 1, "buy": 1, "kaufen": 1, "kauf": 1,
    "bullisch": 1, "moon": 1, "pump": 1, "yolo": 1, "fomo": 1, "hodl": 1, "btfd": 1,
    "rocket": 1, "squeeze": 1, "lfg": 1, "hoch": 1, "lambo": 1, "🚀": 1, "green": 1,
    "grün": 1, "profit": 1, "gewinn": 1, "rally": 1, "boom": 1,
    # negative
    "short": -1, "put": -1, "bear": -1, "bearish": -1, "sell": -1, "verkaufen": -1,
    "bärisch": -1, "dip": -1, "dump": -1, "dumping": -1, "fud": -1, "crash": -1,
    "bagholder": -1, "rekt": -1, "down": -1, "fall": -1, "sink": -1, "stupid": -1,
    "schlecht": -1, "red": -1, "rot": -1, "loss": -1, "verlust": -1, "collapse": -1,
    "baisse": -1,
}

def _load_keywords_from_file(path: str | Path | None = None, keywords: dict[str, int] | None = None) -> None:
    if keywords:
        for word, score in keywords.items():
            try:
                norm = normalize_token(str(word).lower())
                KEYWORD_SCORES.setdefault(norm, int(score))
            except Exception:
                continue
    root = Path(__file__).resolve().parents[1]
    candidates = (
        [Path(path)] if path else [
            root / "data" / "sentiment_keywords.json",
            root / "data" / "sentiment_keywords.yaml",
            root / "data" / "sentiment_keywords.yml",
        ]
    )
    for cand in candidates:
        if not cand.exists():
            continue
        try:
            with cand.open("r", encoding="utf-8") as fh:
                if cand.suffix == ".json":
                    data = json.load(fh)
                elif cand.suffix in {".yaml", ".yml"} and yaml:
                    data = yaml.safe_load(fh)
                else:
                    logger.warning("Unsupported sentiment keyword file format: %s", cand)
                    data = {}
            for word, score in (data or {}).items():
                try:
                    norm = normalize_token(str(word).lower())
                    KEYWORD_SCORES.setdefault(norm, int(score))
                except Exception:
                    continue
        except Exception as exc:
            logger.warning("Could not load sentiment keywords from %s: %s", cand, exc)
        break  # nur die erste existierende Datei nutzen

# Konfig & Keywords laden, dann Varianten ableiten
_load_sentiment_config()
_load_keywords_from_file()
for _w, _s in list(KEYWORD_SCORES.items()):
    for _i, _m in INTENSITY_WEIGHTS.items():
        KEYWORD_SCORES[f"{_i} {_w}"] = _s * _m
    KEYWORD_SCORES[f"not_{_w}"] = -_s

def apply_negation(text: str) -> str:
    """Negations-Faltung: markiert Keyword im Fenster von 2 Tokens mit 'not_'."""
    words = text.split()
    result: list[str] = []
    i = 0
    while i < len(words):
        w = words[i]
        if w in NEGATION_MARKERS:
            target = None
            for off in (1, 2):
                j = i + off
                if j < len(words):
                    cand = words[j]
                    if cand in KEYWORD_SCORES or cand in INTENSITY_WEIGHTS:
                        target = j
                        break
            if target is not None:
                result.extend(words[i + 1:target])
                result.append(f"not_{words[target]}")
                i = target + 1
                continue
            i += 1
            continue
        result.append(w)
        i += 1
    return " ".join(result)

# ---------------------------------------------------------------------------
# FinBERT (bevorzugt) + VADER Fallback
# ---------------------------------------------------------------------------
@dataclass
class _Scores:
    compound: float
    pos: float
    neg: float
    neu: float
    label: str

class FinBertAdapter:
    """Leichter Wrapper für ProsusAI/finbert (lokal > Cache)."""
    def __init__(self, model_name: str = "ProsusAI/finbert", local_dir: Optional[str] = "models/finbert") -> None:
        self.model_name = model_name
        self.local_dir = local_dir
        self._model = None
        self._tokenizer = None

    def _resolve_ref(self) -> str:
        if self.local_dir and os.path.isdir(self.local_dir):
            return self.local_dir
        return self.model_name

    @property
    def model(self):  # pragma: no cover - heavy
        if self._model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            ref = self._resolve_ref()
            local_only = os.path.isdir(ref)
            self._tokenizer = AutoTokenizer.from_pretrained(ref, local_files_only=local_only)
            self._model = AutoModelForSequenceClassification.from_pretrained(ref, local_files_only=local_only)
        return self._model

    @property
    def tokenizer(self):  # pragma: no cover - heavy
        if self._tokenizer is None:
            _ = self.model
        return self._tokenizer

    def __call__(self, text: str | list[str], truncation: bool = True, max_length: int = 512):
        import torch  # pragma: no cover - heavy
        tok = self.tokenizer
        mdl = self.model

        if isinstance(text, str):
            inputs = tok(text, return_tensors="pt", truncation=truncation, max_length=max_length)
            with torch.no_grad():
                outputs = mdl(**inputs)
            probs = outputs.logits.softmax(dim=-1)[0]
            idx = int(torch.argmax(probs).item())
            label = mdl.config.id2label[idx]
            score = float(probs[idx].item())
            return [{"label": label, "score": score}]

        inputs = tok(text, return_tensors="pt", truncation=truncation, max_length=max_length, padding=True)
        with torch.no_grad():
            outputs = mdl(**inputs)
        probs = outputs.logits.softmax(dim=-1)
        results = []
        for i in range(probs.shape[0]):
            idx = int(torch.argmax(probs[i]).item())
            label = mdl.config.id2label[idx]
            score = float(probs[i, idx].item())
            results.append({"label": label, "score": score})
        return results


class BertSentiment:
    """Thin wrapper for backward compatibility in tests."""

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        local_dir: Optional[str] = "models/finbert",
    ) -> None:
        self._adapter = FinBertAdapter(model_name, local_dir)

    def __call__(self, text, truncation: bool = True, max_length: int = 512):
        return self._adapter(text, truncation=truncation, max_length=max_length)

class _Engine:
    """Singleton-Engine: FinBERT wenn möglich, sonst VADER."""
    _pipe = None
    _vader = None
    _init_done = False

    @classmethod
    def init(cls) -> None:
        if cls._init_done:
            return
        cls._init_done = True

        backend = (settings.SENTIMENT_BACKEND or "finbert").lower()
        want_finbert = backend in ("finbert", "auto", "true", "1")

        if want_finbert and _transformers_available():
            try:
                cls._pipe = BertSentiment(
                    "ProsusAI/finbert", local_dir=os.getenv("FINBERT_LOCAL_DIR", "models/finbert")
                )
                logger.info("Sentiment: FinBERT adapter ready")
            except Exception as e:
                logger.warning(f"FinBERT adapter init failed: {e}")
                cls._pipe = None

        if cls._pipe is None:
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                v = SentimentIntensityAnalyzer()
                v.lexicon.update({
                    "🚀": 2.5, "💎🙌": 2.5, "yolo": 2.0, "bagholder": -2.0, "paper hands": -2.0,
                    "long": 1.2, "short": -1.4, "calls": 1.2, "puts": -1.4,
                    "bullish": 1.5, "bearish": -1.5,
                })
                cls._vader = v
                logger.info("Sentiment: VADER ready (fallback)")
            except Exception as e:
                logger.error(f"VADER init failed: {e}")
                cls._vader = None

    @classmethod
    def backend_name(cls) -> str:
        cls.init()
        if cls._pipe is not None:
            return "finbert"
        if cls._vader is not None:
            return "vader"
        return "keyword"

# ---------------------------------------------------------------------------
# Öffentliche API (beibehaltener Funktionssatz)
# ---------------------------------------------------------------------------
def get_backend_name() -> str:
    """'finbert' | 'vader' | 'keyword'"""
    return _Engine.backend_name()

def analyze_sentiment_bert(text: str) -> float:
    """Einzelwert via FinBERT ([-1..+1]); wirft nicht."""
    global _bert_analyzer
    if _bert_analyzer is None:
        _Engine._init_done = False
        _Engine.init()
        _bert_analyzer = _Engine._pipe
    if _bert_analyzer is None:
        return 0.0
    try:
        out = _bert_analyzer(text)
        entries = _normalize_hf_output(out)
        if not entries:
            return 0.0
        scalar = _scores_to_scalar(entries)
        return _blend_model_scores(text, scalar)
    except Exception as e:
        logger.warning(f"FinBERT inference failed: {e}")
        return 0.0

def analyze_sentiment(text: str) -> float:
    """Skalarer Score in [-1, +1]; Reihenfolge: FinBERT → VADER → Keywords."""
    t = (text or "").strip()
    if not t:
        return 0.0
    if getattr(settings, "USE_BERT_SENTIMENT", False):
        val = analyze_sentiment_bert(t)
        if val != 0.0:
            return float(val)
    # 1) FinBERT
    if _Engine.backend_name() == "finbert":
        val = analyze_sentiment_bert(t)
        if val != 0.0:
            return float(val)

    # 2) VADER
    if _Engine._vader is not None:
        try:
            s = _Engine._vader.polarity_scores(t)
            return float(s["compound"])
        except Exception:
            pass

    # 3) Keyword-Heuristik
    _log_keyword_hint("Using keyword-based sentiment analysis (fallback)")
    val = _keyword_score_cached(_preprocess(t))
    return float(val if val is not None else 0.0)

def analyze_sentiment_batch(texts: list[str]) -> list[float]:
    """Batch-Score; nutzt FinBERT Batch, sonst VADER/Keywords zeilenweise."""
    _Engine.init()
    if not texts:
        return []
    if _Engine._pipe is not None:
        try:
            out = _Engine._pipe(texts)
            if not isinstance(out, (list, tuple)):
                out = [out]
            scores = []
            for txt, data in zip(texts, out):  # noqa: B905
                entries = _normalize_hf_output(data)
                scalar = _scores_to_scalar(entries)
                scores.append(_blend_model_scores(txt, scalar))
            return scores
        except Exception as e:
            logger.warning(f"FinBERT batch inference failed: {e}")
    # VADER / Keywords
    return [analyze_sentiment(t) for t in texts]

# ---------------------------------------------------------------------------
# Keyword-Scorer (Fallback)
# ---------------------------------------------------------------------------
def _keyword_score(text: str) -> float | None:
    text = apply_negation(text)
    tokens = text.split()
    score = 0.0
    matched = False
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        mult = 1
        if tok in INTENSITY_WEIGHTS and i + 1 < len(tokens):
            mult = INTENSITY_WEIGHTS[tok]
            i += 1
            tok = tokens[i]
        val = KEYWORD_SCORES.get(tok)
        if val:
            matched = True
            score += val * mult
        i += 1
    return score if matched else None

_keyword_score_cached = lru_cache(maxsize=2048)(_keyword_score)


def _normalize_hf_output(raw) -> list[dict[str, float]]:
    """Flatten HuggingFace-style outputs to ``[{label, score}, ...]``."""

    if isinstance(raw, dict):
        if "label" in raw and "score" in raw:
            try:
                score = float(raw["score"])
            except Exception:
                score = 0.0
            return [{"label": str(raw["label"]), "score": score}]
        return []

    if isinstance(raw, (list, tuple)):
        entries: list[dict[str, float]] = []
        for item in raw:
            entries.extend(_normalize_hf_output(item))
        return entries

    return []


def _scores_to_scalar(entries: list[dict[str, float]]) -> float:
    """Convert classifier label scores into ``[-1, 1]`` sentiment scalar."""

    if not entries:
        return 0.0

    scores: dict[str, float] = {}
    for entry in entries:
        label = str(entry.get("label", "")).strip().lower()
        if not label:
            continue
        try:
            score = float(entry.get("score", 0.0))
        except Exception:
            score = 0.0
        current = scores.get(label)
        if current is None or score > current:
            scores[label] = score

    pos = scores.get("positive") or scores.get("bullish") or 0.0
    neg = scores.get("negative") or scores.get("bearish") or 0.0

    scalar = pos - neg
    return float(max(-1.0, min(1.0, scalar)))


def _scaled_keyword_boost(text: str) -> float:
    """Return a small boost based on keyword heuristics for model outputs."""

    raw = _keyword_score_cached(_preprocess(text))
    if raw is None or raw == 0:
        return 0.0
    boost = float(raw) / 4.0
    return max(-0.6, min(0.6, boost))


def _blend_model_scores(text: str, scalar: float) -> float:
    """Combine model scalar with fallback heuristics/VADER and clamp to [-1, 1]."""

    score = float(scalar)
    if _Engine._vader is not None:
        try:
            vader = float(_Engine._vader.polarity_scores(text).get("compound", 0.0))
        except Exception:
            vader = 0.0
        # Wenn FinBERT unsicher ist (<|0.2|), Gewichtung mit VADER.
        if abs(score) < 0.2 and vader:
            score = 0.6 * score + 0.4 * vader
    score += _scaled_keyword_boost(text)
    return float(max(-1.0, min(1.0, score)))

# ---------------------------------------------------------------------------
# Aggregation / Utilities
# ---------------------------------------------------------------------------
def _post_weight(upvotes: int | float = 0, num_comments: int | float = 0) -> float:
    """Log-gewichtung (robust gegen 0): 1 + log10(ups+1) + 0.2*log10(com+1)"""
    try:
        u = max(0.0, float(upvotes))
        c = max(0.0, float(num_comments))
    except Exception:
        u, c = 0.0, 0.0
    return 1.0 + math.log10(1.0 + u) + 0.2 * math.log10(1.0 + c)

def _ensure_datetime_series(s: "pd.Series", tz: str = "Europe/Berlin") -> "pd.Series":
    s = pd.to_datetime(s, errors="coerce", utc=True)  # type: ignore
    return s.dt.tz_convert(ZoneInfo(tz))  # type: ignore

def _safe_float_series(s: "pd.Series") -> "pd.Series":
    return pd.to_numeric(s, errors="coerce").astype(float)  # type: ignore

def compute_daily_sentiment(
    df: "pd.DataFrame",
    *,
    text_col: str = "text",
    ts_col: str = "created_at",
    ticker_col: str = "ticker",
    upvotes_col: str = "upvotes",
    comments_col: str = "num_comments",
) -> "pd.DataFrame":
    """
    Liefert pro Tag (Europe/Berlin) & Ticker den gewichteten Sentiment-Mittelwert.
    Rückgabe: DataFrame[date, ticker, posts, wgt_sum, wgt_mean]
    """
    if pd is None:  # pragma: no cover
        raise RuntimeError("pandas not available")
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["date", "ticker", "posts", "wgt_sum", "wgt_mean"])

    ts = _ensure_datetime_series(df[ts_col], tz="Europe/Berlin")
    date = ts.dt.normalize()
    tmp = df.copy()

    scores = tmp[text_col].astype(str).map(analyze_sentiment)
    ups = _safe_float_series(tmp.get(upvotes_col, 0))
    com = _safe_float_series(tmp.get(comments_col, 0))
    weights = [ _post_weight(u, c) for u, c in zip(ups.fillna(0.0), com.fillna(0.0)) ]

    tmp["_date"] = date
    tmp["_score"] = pd.to_numeric(scores, errors="coerce").fillna(0.0)
    tmp["_w"] = pd.to_numeric(pd.Series(weights, index=tmp.index), errors="coerce").fillna(1.0)

    g = tmp.groupby([ticker_col, "_date"], as_index=False).agg(
        posts=(text_col, "count"),
        wgt_sum=("_score", lambda s: float((s * tmp.loc[s.index, "_w"]).sum())),
        w_sum=("_w", "sum"),
    )
    g["wgt_mean"] = (g["wgt_sum"] / g["w_sum"]).replace([float("inf"), -float("inf")], 0.0).fillna(0.0)

    out = g.rename(columns={ticker_col: "ticker", "_date": "date"})[["date", "ticker", "posts", "wgt_sum", "wgt_mean"]]
    out["posts"] = pd.to_numeric(out["posts"], errors="coerce").fillna(0).astype("Int64")
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)

def aggregate_sentiment_by_ticker(ticker_texts: dict[str, Iterable[dict]]) -> dict[str, float]:
    """Mittelwert je Ticker; nutzt analyze_sentiment() inkl. Fallback-Reihenfolge."""
    result: dict[str, float] = {}
    for ticker, entries in ticker_texts.items():
        entries = list(entries)
        if not entries:
            result[ticker] = 0.0
            continue
        tot = 0.0
        n = 0
        for entry in entries:
            text = entry.get("text", "") if isinstance(entry, dict) else str(entry)
            tot += float(analyze_sentiment(text) or 0.0)
            n += 1
        result[ticker] = (tot / n) if n else 0.0
    return result

def derive_recommendation(score: float) -> str:
    """Sehr grobe Einordnung: >0 → Buy, <0 → Sell, sonst Hold."""
    if score > 0:
        return "Buy"
    if score < 0:
        return "Sell"
    return "Hold"

__all__ = [
    "get_backend_name",
    "analyze_sentiment",
    "analyze_sentiment_batch",
    "analyze_sentiment_bert",
    "aggregate_sentiment_by_ticker",
    "compute_daily_sentiment",
    "derive_recommendation",
]
