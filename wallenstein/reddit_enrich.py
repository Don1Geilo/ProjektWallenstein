from __future__ import annotations

import math
from typing import Dict, List, Tuple, Optional

import duckdb
import pandas as pd

from .db_schema import validate_df
from .sentiment import analyze_sentiment


# ---------- helpers ----------

def _to_str_id(value) -> Optional[str]:
    """Reddit-ID immer als String speichern (keine Base36->Int-Konvertierung!)."""
    if value is None:
        return None
    try:
        s = str(value).strip()
        return s if s else None
    except Exception:
        return None


def _to_upvotes(value) -> int:
    """Upvotes robust parsen + clampen (für log1p)."""
    try:
        up = int(value)
    except Exception:
        up = 0
    return max(0, min(10_000_000, up))


def _to_ts_utc_naive(value) -> Optional[pd.Timestamp]:
    """Nach UTC parsen und tz-naiv zurückgeben (kompatibel zu DuckDB TIMESTAMP)."""
    try:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return None
    if pd.isna(ts):
        return None
    # als naive UTC (ohne tz) speichern
    return ts.tz_convert(None)


# ---------- main funcs ----------

def enrich_reddit_posts(
    con: duckdb.DuckDBPyConnection,
    posts_by_ticker: dict[str, list[dict]],
) -> int:
    """Enrich und persistiere Reddit-Posts in reddit_enriched.
    Gibt die Anzahl der eingefügten Zeilen zurück.
    """
    rows: list[dict] = []

    for ticker, posts in (posts_by_ticker or {}).items():
        for post in posts or []:
            post_id = _to_str_id(post.get("id") or post.get("name") or post.get("link_id"))
            created = _to_ts_utc_naive(post.get("created_utc"))
            if not post_id or created is None:
                continue

            text = str(post.get("text") or "")[:4096]
            upvotes = _to_upvotes(post.get("upvotes", post.get("score")))
            sent = analyze_sentiment(text)  # darf None sein
            weighted = None if sent is None else float(sent) * math.log1p(upvotes)

            rows.append(
                {
                    "id": post_id,
                    "ticker": ticker.upper(),
                    "created_utc": created.to_pydatetime(),  # duckdb py binding mag py datetime
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

    # id+ticker als Schlüssel: delete+insert (deterministisch, upsert-sicher)
    key_pairs = df[["id", "ticker"]].values.tolist()
    con.executemany("DELETE FROM reddit_enriched WHERE id = ? AND ticker = ?", key_pairs)

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
        f"INSERT INTO reddit_enriched ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
        df[cols].values.tolist(),
    )
    return len(df)


def compute_reddit_trends(con: duckdb.DuckDBPyConnection) -> int:
    """Aggregiere tägliche Reddit-Trends in reddit_trends."""
    # DuckDB akzeptiert INSERT OR REPLACE – falls nicht, alternativ: DELETE+INSERT via Temp-Table
    result = con.execute(
        """
        INSERT OR REPLACE INTO reddit_trends
        SELECT
            DATE_TRUNC('day', created_utc) AS date,
            ticker,
            COUNT(*) AS mentions,
            AVG(upvotes) AS avg_upvotes,
            COUNT(*) * AVG(upvotes) AS hotness
        FROM reddit_enriched
        WHERE sentiment_dict IS NOT NULL
        GROUP BY 1, 2
        """
    )
    # rowcount kann bei DuckDB 0 sein, obwohl Daten geschrieben wurden; nicht kritisch
    return int(getattr(result, "rowcount", 0) or 0)


def compute_returns(
    con: duckdb.DuckDBPyConnection,
    horizon_days: tuple[int, ...] = (1, 3, 7),
) -> int:
    """Berechne Forward-Returns (1d/3d/7d) je Post in reddit_enriched."""
    if not horizon_days:
        return 0

    df_posts = con.execute(
        "SELECT id, ticker, created_utc FROM reddit_enriched"
    ).fetch_df()
    if df_posts.empty:
        return 0

    df_prices = con.execute("SELECT date, ticker, close FROM prices").fetch_df()
    if df_prices.empty:
        return 0

    df_prices["date"] = pd.to_datetime(df_prices["date"])
    df_posts["created_utc"] = pd.to_datetime(df_posts["created_utc"]).dt.normalize()

    updated = 0

    # Gruppiere Preise je Ticker für effiziente Suche
    for tkr, grp in df_prices.groupby("ticker"):
        grp = grp.sort_values("date").reset_index(drop=True)
        # Posts für diesen Ticker:
        posts_t = df_posts[df_posts["ticker"] == tkr]
        if posts_t.empty:
            continue

        dates = grp["date"].values
        closes = grp["close"].values

        for r in posts_t.itertuples(index=False):
            # Basis: erster Handelstag >= created_utc
            idx0 = grp["date"].searchsorted(r.created_utc, side="left")
            if idx0 >= len(grp):
                continue
            base_close = float(closes[idx0])

            ret_vals: list[Optional[float]] = []
            for h in horizon_days:
                idxN = idx0 + h  # N Handelstage weiter (nicht Kalendertage)
                if idxN >= len(grp):
                    ret_vals.append(None)
                else:
                    ret_vals.append((float(closes[idxN]) - base_close) / base_close)

            set_clause = ", ".join(f"return_{h}d = ?" for h in horizon_days)
            con.execute(
                f"UPDATE reddit_enriched SET {set_clause} WHERE id = ? AND ticker = ?",
                ret_vals + [r.id, tkr],
            )
            updated += 1

    return updated


__all__ = ["enrich_reddit_posts", "compute_reddit_trends", "compute_returns"]
