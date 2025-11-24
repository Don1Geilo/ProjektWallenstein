"""Utilities for discovering tickers mentioned in Reddit posts."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable

log = logging.getLogger(__name__)


CASHTAG_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])[#$]\s*([A-Za-z][A-Za-z0-9\.\-]{0,9})",
    re.IGNORECASE,
)

# Match plain ticker style words such as TSLA, NVDA, RHM.DE
PLAIN_TICKER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])([A-Z][A-Z0-9]{1,4}(?:\.[A-Z0-9]{1,3})?)(?![A-Za-z0-9])"
)

# Very common short words that are frequently capitalised in Reddit posts but are
# not tickers.  This helps to avoid unnecessary lookups against yfinance.
SYMBOL_STOPWORDS: set[str] = {
    "A",
    "AN",
    "AI",
    "ALL",
    "AND",
    "API",
    "AS",
    "ASAP",
    "ATH",
    "B",
    "BBQ",
    "CEO",
    "CFO",
    "CPI",
    "CPU",
    "DD",
    "DM",
    "DNA",
    "DT",
    "ES",
    "ETF",
    "EU",
    "EV",
    "FED",
    "FOMO",
    "GDP",
    "IF",
    "IMO",
    "IPO",
    "IRS",
    "IT",
    "LOL",
    "MOON",
    "NEED",
    "OTC",
    "PM",
    "PUMP",
    "QQQ",
    "SEC",
    "SOON",
    "SPAC",
    "THE",
    "TLDR",
    "TO",
    "USA",
    "USD",
    "WSB",
    "YOLO",
}

# Corporate suffixes that can be stripped from long names when generating alias
# variants.
CORPORATE_SUFFIXES = {
    "ab",
    "ag",
    "asa",
    "class a",
    "class b",
    "class c",
    "co",
    "co.",
    "company",
    "corp",
    "corp.",
    "corporation",
    "group",
    "holding",
    "holdings",
    "inc",
    "inc.",
    "incorporated",
    "limited",
    "ltd",
    "nv",
    "oyj",
    "plc",
    "sa",
    "se",
    "spa",
}


@dataclass(frozen=True)
class TickerMetadata:
    """Metadata returned for newly discovered tickers."""

    symbol: str
    aliases: set[str]


def _clean_symbol(raw: str) -> str | None:
    symbol = (raw or "").strip().upper()
    if not symbol:
        return None
    symbol = symbol.strip("$#.,:;!?")
    if not symbol or len(symbol) > 10:
        return None
    if symbol in SYMBOL_STOPWORDS:
        return None
    if not any(ch.isalpha() for ch in symbol):
        return None
    if not re.fullmatch(r"[A-Z0-9]+(?:\.[A-Z0-9]+)?", symbol):
        return None
    return symbol


def _extract_symbol_sets(text: str) -> tuple[set[str], set[str]]:
    """Return (plain_hits, cashtag_hits) for ``text``."""

    if not text:
        return set(), set()

    cashtags = {
        sym
        for sym in (_clean_symbol(match) for match in CASHTAG_PATTERN.findall(text))
        if sym
    }

    plain_hits = {
        sym
        for sym in (_clean_symbol(match) for match in PLAIN_TICKER_PATTERN.findall(text))
        if sym
    }

    plain_hits |= cashtags
    return plain_hits, cashtags


def extract_candidate_symbols(texts: Iterable[str]) -> dict[str, dict[str, int]]:
    """Analyse ``texts`` and count potential ticker occurrences."""

    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "cashtag": 0})
    for text in texts:
        plain_hits, cashtag_hits = _extract_symbol_sets(text)
        for sym in plain_hits:
            stats[sym]["total"] += 1
        for sym in cashtag_hits:
            stats[sym]["cashtag"] += 1
    return stats


def _alias_variants(name: str) -> set[str]:
    name = " ".join(str(name or "").strip().split())
    if not name:
        return set()
    lower = name.lower()
    simplified = re.sub(r"[^a-z0-9& ]+", " ", lower)
    simplified = " ".join(simplified.split())

    variants: set[str] = set()
    if simplified:
        variants.add(simplified)
        if "&" in simplified:
            variants.add(simplified.replace("&", "and"))

    tokens = simplified.split()
    while tokens and tokens[-1] in CORPORATE_SUFFIXES:
        tokens = tokens[:-1]
        variant = " ".join(tokens)
        if variant:
            variants.add(variant)

    return {v for v in variants if len(v) >= 3}


def fetch_ticker_metadata(symbol: str) -> TickerMetadata | None:
    """Validate ``symbol`` via yfinance and return name based aliases."""

    try:
        import yfinance as yf
    except Exception as exc:  # pragma: no cover - yfinance missing in env
        raise RuntimeError("yfinance is required for ticker discovery") from exc

    ticker = yf.Ticker(symbol)

    info: dict[str, str] = {}
    try:
        info = ticker.get_info() or {}
    except Exception as exc:  # pragma: no cover - network hiccup
        log.debug("yfinance.get_info failed for %s: %s", symbol, exc)
        info = {}

    alias_candidates = [
        info.get(key)
        for key in ("shortName", "longName", "name", "displayName")
        if isinstance(info.get(key), str) and info.get(key)
    ]

    if not alias_candidates:
        # Fallback: check fast_info/price data to ensure the symbol exists.
        fast_info = None
        try:
            fast_info = ticker.fast_info  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - robustness
            log.debug("yfinance.fast_info failed for %s: %s", symbol, exc)

        price_value = None
        if isinstance(fast_info, dict):
            for key in ("last_price", "lastPrice", "regularMarketPrice", "regular_market_price"):
                if key in fast_info and fast_info[key] is not None:
                    price_value = fast_info[key]
                    break
        else:
            for attr in ("last_price", "lastPrice", "regularMarketPrice", "regular_market_price"):
                price_value = getattr(fast_info, attr, None)
                if price_value is not None:
                    break

        if price_value is None:
            try:
                hist = ticker.history(period="5d")
            except Exception as exc:  # pragma: no cover - network failure
                log.debug("yfinance.history failed for %s: %s", symbol, exc)
                hist = None
            if hist is None or getattr(hist, "empty", True):
                return None

    aliases = set()
    for alias in alias_candidates:
        aliases.update(_alias_variants(alias))

    return TickerMetadata(symbol=symbol.upper(), aliases=aliases)


def discover_new_tickers(
    texts: Iterable[str],
    known: Iterable[str] | None = None,
    *,
    fetch_metadata: Callable[[str], TickerMetadata | None] | None = None,
    min_plain_mentions: int = 2,
) -> dict[str, TickerMetadata]:
    """Return mapping of newly discovered tickers to their metadata."""

    fetcher = fetch_metadata or fetch_ticker_metadata
    known_set = {sym.upper() for sym in (known or [])}
    stats = extract_candidate_symbols(texts)

    discovered: dict[str, TickerMetadata] = {}
    for symbol, counts in stats.items():
        if symbol in known_set:
            continue
        if counts["cashtag"] == 0 and counts["total"] < min_plain_mentions:
            continue
        try:
            meta = fetcher(symbol)
        except RuntimeError as exc:  # pragma: no cover - missing dependency
            raise
        except Exception as exc:  # pragma: no cover - robustness
            log.debug("Metadata lookup failed for %s: %s", symbol, exc)
            continue
        if not meta:
            continue
        discovered[symbol] = meta

    return discovered

