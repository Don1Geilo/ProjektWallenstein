from __future__ import annotations

import logging
from typing import Iterable

import duckdb
import pandas as pd
from scipy.stats import pearsonr, spearmanr


log = logging.getLogger(__name__)


def _prepare_frames(
    con: duckdb.DuckDBPyConnection, tickers: Iterable[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    symbols = {str(t).upper() for t in tickers if t}
    if not symbols:
        return pd.DataFrame(), pd.DataFrame()

    placeholders = ",".join("?" for _ in symbols)

    try:
        df_sent = con.execute(
            f"""
            SELECT date, ticker, sentiment_weighted AS sentiment
            FROM reddit_sentiment_daily
            WHERE ticker IN ({placeholders})
              AND sentiment_weighted IS NOT NULL
            """,
            list(symbols),
        ).fetchdf()
    except duckdb.Error:
        df_sent = pd.DataFrame(columns=["date", "ticker", "sentiment"])

    try:
        df_prices = con.execute(
            f"""
            SELECT date, ticker, close
            FROM prices
            WHERE ticker IN ({placeholders})
            """,
            list(symbols),
        ).fetchdf()
    except duckdb.Error:
        df_prices = pd.DataFrame(columns=["date", "ticker", "close"])

    return df_sent, df_prices


def compute_price_sentiment_correlations(
    con: duckdb.DuckDBPyConnection,
    tickers: Iterable[str],
    min_samples: int = 5,
) -> dict[str, dict[str, float | int | None]]:
    """Return Pearson/Spearman correlations of returns vs. sentiment per ticker."""

    df_sent, df_prices = _prepare_frames(con, tickers)
    if df_sent.empty or df_prices.empty:
        return {}

    df_sent["date"] = pd.to_datetime(df_sent["date"], errors="coerce")
    df_prices["date"] = pd.to_datetime(df_prices["date"], errors="coerce")
    df_prices = df_prices.sort_values(["ticker", "date"]).copy()
    df_prices["return"] = df_prices.groupby("ticker")["close"].pct_change(fill_method=None)

    results: dict[str, dict[str, float | int | None]] = {}
    for ticker, grp in df_prices.groupby("ticker"):
        sent = df_sent[df_sent["ticker"].str.upper() == str(ticker).upper()]
        if sent.empty:
            continue

        merged = pd.merge(
            grp[["date", "return"]],
            sent[["date", "sentiment"]],
            on="date",
            how="inner",
        ).dropna()

        if len(merged) < min_samples:
            continue

        try:
            pearson_val, _ = pearsonr(merged["return"], merged["sentiment"])
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("Pearson correlation failed for %s: %s", ticker, exc)
            pearson_val = None

        try:
            spearman_val, _ = spearmanr(merged["return"], merged["sentiment"])
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("Spearman correlation failed for %s: %s", ticker, exc)
            spearman_val = None

        results[str(ticker).upper()] = {
            "pearson": float(pearson_val) if pearson_val is not None else None,
            "spearman": float(spearman_val) if spearman_val is not None else None,
            "samples": int(len(merged)),
        }

    return results


__all__ = ["compute_price_sentiment_correlations"]

