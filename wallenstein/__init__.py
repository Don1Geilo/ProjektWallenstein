"""Wallenstein package exposing utility functions."""

from .sentiment import (
    aggregate_sentiment_by_ticker,
    analyze_sentiment,
    derive_recommendation,
)

# Ensure ``pytest`` is available as a built-in for tests that expect it
try:  # pragma: no cover - only relevant during testing
    import builtins
    import pytest

    builtins.pytest = pytest
except Exception:  # pragma: no cover - ignore if pytest missing
    pass

__all__ = [
    "aggregate_sentiment_by_ticker",
    "analyze_sentiment",
    "derive_recommendation",
]
