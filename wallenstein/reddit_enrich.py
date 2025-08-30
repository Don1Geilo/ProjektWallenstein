from __future__ import annotations

import math

import duckdb
import pandas as pd

from .db_schema import validate_df
from .sentiment import analyze_sentiment


def enrich_reddit_posts(
    con: duckdb.DuckDBPyConnection,
    posts: dict[str, list[dict]],
    tickers: list[str],
) -> None:
    """Enrich raw Reddit posts and store them in ``reddit_enriched``.

    Parameters
    ----------
    con:
        Active DuckDB connection.
    posts:
        Mapping of ticker to a list of post dictionaries as returned by
        :func:`update_reddit_data`.
    tickers:
        List of tickers to consider. The mapping may contain more keys but only
        these are processed.
    """

    rows: list[dict] = []
    for ticker in tickers:
        for post in posts.get(ticker, []):
            raw_id = post.get("id")
            try:
                post_id = int(raw_id, 36)
            except Exception:
                # Skip comments or malformed IDs which cannot be converted
                continue
            created = pd.to_datetime(post.get("created_utc"), utc=True)
            text = str(post.get("text", ""))
            upvotes = int(post.get("upvotes") or 0)
            sent = analyze_sentiment(text)
            weighted = None if sent is None else sent * math.log(upvotes + 1)
            rows.append(
                {
                    "id": post_id,
                    "ticker": ticker,
                    "created_utc": created,
                    "text": text,
                    "upvotes": upvotes,
                    "sentiment_dict": sent,
                    "sentiment_weighted": weighted,
                    "sentiment_ml": None,
                    "return_1d": None,
                    "return_3d": None,
                    "return_7d": None,
                }
            )

    if not rows:
        return

    df = pd.DataFrame(rows)
    validate_df(df, "reddit_enriched")

    ids = df[["id", "ticker"]].values.tolist()
    if ids:
        placeholders = ",".join("(?, ?)" for _ in ids)
        flat = [item for pair in ids for item in pair]
        existing = set(
            con.execute(
                f"SELECT id, ticker FROM reddit_enriched WHERE (id, ticker) IN ({placeholders})",
                flat,
            ).fetchall()
        )
        if existing:
            mask = ~df.apply(lambda r: (int(r["id"]), r["ticker"]) in existing, axis=1)
            df = df.loc[mask]
    if df.empty:
        return

    con.register("df_enriched", df)
    con.execute(
        """
        INSERT INTO reddit_enriched (
            id, ticker, created_utc, text, upvotes,
            sentiment_dict, sentiment_weighted, sentiment_ml,
            return_1d, return_3d, return_7d
        )
        SELECT id, ticker, created_utc, text, upvotes,
               sentiment_dict, sentiment_weighted, sentiment_ml,
               return_1d, return_3d, return_7d
        FROM df_enriched
        """
    )


def compute_reddit_trends(con: duckdb.DuckDBPyConnection) -> None:
    """Aggregate daily Reddit trends into ``reddit_trends`` table."""

    con.execute(
        """
        INSERT OR REPLACE INTO reddit_trends
        SELECT date, ticker, mentions, avg_upvotes, hotness
        FROM (
            SELECT DATE_TRUNC('day', created_utc) AS date,
                   ticker,
                   COUNT(*) AS mentions,
                   AVG(upvotes) AS avg_upvotes,
                   COUNT(*) * AVG(upvotes) AS hotness
            FROM reddit_enriched
            WHERE sentiment_dict IS NOT NULL
            GROUP BY date, ticker
        )
        """
    )


def compute_returns(
    con: duckdb.DuckDBPyConnection,
    horizon_days: tuple[int, ...] = (1, 3, 7),
) -> None:
    """Compute forward returns for posts in ``reddit_enriched``."""

    if not horizon_days:
        return

    df_posts = con.execute(
        "SELECT id, ticker, created_utc FROM reddit_enriched"
    ).fetch_df()
    if df_posts.empty:
        return

    df_prices = con.execute(
        "SELECT date, ticker, close FROM prices"
    ).fetch_df()
    if df_prices.empty:
        return

    df_prices["date"] = pd.to_datetime(df_prices["date"])

    for row in df_posts.itertuples(index=False):
        prices_t = df_prices[df_prices["ticker"] == row.ticker].sort_values("date")
        base = prices_t[
            prices_t["date"] >= pd.to_datetime(row.created_utc).normalize()
        ].head(1)
        if base.empty:
            continue
        base_date = base["date"].iloc[0]
        base_close = base["close"].iloc[0]
        for h in horizon_days:
            target = prices_t[prices_t["date"] >= base_date + pd.Timedelta(days=h)].head(1)
            if target.empty:
                ret = None
            else:
                ret = (target["close"].iloc[0] - base_close) / base_close
            con.execute(
                f"UPDATE reddit_enriched SET return_{h}d = ? WHERE id = ? AND ticker = ?",
                [ret, row.id, row.ticker],
            )
