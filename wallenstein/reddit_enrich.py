from __future__ import annotations

import math

import duckdb
import pandas as pd

from .db_schema import validate_df
from .sentiment import analyze_sentiment


def _to_str_id(value) -> str | None:
    try:
        return str(int(str(value), 36))
    except Exception:
        return None


def _to_upvotes(value) -> int:
    try:
        up = int(value)
    except Exception:
        up = 0
    return max(0, min(10_000_000, up))


def _to_ts(value) -> pd.Timestamp | None:
    try:
        ts = pd.to_datetime(value, utc=True)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return ts


def enrich_reddit_posts(
    con: duckdb.DuckDBPyConnection,
    posts_by_ticker: dict[str, list[dict]],
) -> int:
    """Enrich and persist Reddit posts.

    Returns the number of inserted rows in ``reddit_enriched``.
    """

    rows: list[dict] = []
    for ticker, posts in posts_by_ticker.items():
        for post in posts:
            post_id = _to_str_id(post.get("id"))
            created = _to_ts(post.get("created_utc"))
            if not post_id or created is None:
                continue
            text = str(post.get("text", ""))
            upvotes = _to_upvotes(post.get("upvotes"))
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
        return 0

    df = pd.DataFrame(rows).drop_duplicates(subset=["id", "ticker"])
    validate_df(df, "reddit_enriched")

    id_pairs = df[["id", "ticker"]].values.tolist()
    if id_pairs:
        con.executemany(
            "DELETE FROM reddit_enriched WHERE id = ? AND ticker = ?",
            id_pairs,
        )

    cols = [
        "id",
        "ticker",
        "created_utc",
        "text",
        "upvotes",
        "sentiment_dict",
        "sentiment_weighted",
        "sentiment_ml",
        "return_1d",
        "return_3d",
        "return_7d",
    ]
    con.executemany(
        f"""
        INSERT INTO reddit_enriched (
            {', '.join(cols)}
        ) VALUES ({', '.join('?' for _ in cols)})
        """,
        df[cols].values.tolist(),
    )
    return len(df)


def compute_reddit_trends(con: duckdb.DuckDBPyConnection) -> int:
    """Aggregate daily Reddit trends into ``reddit_trends``."""

    result = con.execute(
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
        """,
    )
    return int(result.rowcount)


def compute_returns(
    con: duckdb.DuckDBPyConnection,
    horizon_days: tuple[int, ...] = (1, 3, 7),
) -> int:
    """Compute forward returns for posts in ``reddit_enriched``."""

    if not horizon_days:
        return 0

    df_posts = con.execute(
        "SELECT id, ticker, created_utc FROM reddit_enriched"
    ).fetch_df()
    if df_posts.empty:
        return 0

    df_prices = con.execute(
        "SELECT date, ticker, close FROM prices"
    ).fetch_df()
    if df_prices.empty:
        return 0
    df_prices["date"] = pd.to_datetime(df_prices["date"])

    updated = 0
    for row in df_posts.itertuples(index=False):
        prices_t = df_prices[df_prices["ticker"] == row.ticker].sort_values("date")
        base = prices_t[
            prices_t["date"] >= pd.to_datetime(row.created_utc).normalize()
        ].head(1)
        if base.empty:
            continue
        base_date = base["date"].iloc[0]
        base_close = base["close"].iloc[0]

        ret_vals: list[float | None] = []
        for h in horizon_days:
            target = prices_t[prices_t["date"] >= base_date + pd.Timedelta(days=h)].head(1)
            if target.empty:
                ret = None
            else:
                ret = (target["close"].iloc[0] - base_close) / base_close
            ret_vals.append(ret)

        set_clause = ", ".join(f"return_{h}d = ?" for h in horizon_days)
        con.execute(
            f"UPDATE reddit_enriched SET {set_clause} WHERE id = ? AND ticker = ?",
            ret_vals + [row.id, row.ticker],
        )
        updated += 1
    return updated


__all__ = [
    "enrich_reddit_posts",
    "compute_reddit_trends",
    "compute_returns",
]

