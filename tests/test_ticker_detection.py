from __future__ import annotations

from wallenstein.ticker_detection import (
    TickerMetadata,
    _alias_variants,
    discover_new_tickers,
    extract_candidate_symbols,
)


def test_extract_candidate_symbols_counts_cashtags():
    texts = ["$AAPL to the moon", "I also like TSLA", "#tsla future"]
    stats = extract_candidate_symbols(texts)

    assert stats["AAPL"]["cashtag"] == 1
    assert stats["AAPL"]["total"] == 1
    # TSLA appears twice (once as plain uppercase)
    assert stats["TSLA"]["cashtag"] == 1
    assert stats["TSLA"]["total"] == 2


def test_discover_new_tickers_uses_fetcher(monkeypatch):
    texts = ["$AAPL breakout", "Holding AAPL long term"]

    fetched: list[str] = []

    def fake_fetch(symbol: str) -> TickerMetadata:
        fetched.append(symbol)
        return TickerMetadata(symbol=symbol, aliases={"apple", "apple inc"})

    discovered = discover_new_tickers(texts, known=["NVDA"], fetch_metadata=fake_fetch)

    assert "AAPL" in discovered
    assert fetched == ["AAPL"]
    assert "apple" in discovered["AAPL"].aliases


def test_discover_requires_multiple_plain_mentions():
    texts = ["AAPL might rally"]

    def fail_fetch(symbol: str):  # pragma: no cover - should not be called
        raise AssertionError("fetch should not be triggered")

    discovered = discover_new_tickers(texts, known=[], fetch_metadata=fail_fetch)

    assert discovered == {}


def test_alias_variants_strip_suffixes():
    variants = _alias_variants("Apple Inc.")

    assert "apple" in variants
    assert "apple inc" in variants
    # Ensure suffix removal handled
    assert "apple inc." not in variants


def test_symbol_stopwords_filter_common_words():
    texts = [
        "$THE market is volatile",
        "#if only we knew",
        "$A setup",
        "$B",
        "$AS",
        "$AN",
        "$ES",
        "$NEED",
    ]

    stats = extract_candidate_symbols(texts)

    assert stats == {}
