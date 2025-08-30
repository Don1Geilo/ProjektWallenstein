from __future__ import annotations

import math

import duckdb
import pandas as pd

from .db_schema import validate_df
from .sentiment import analyze_sentiment

# =========================
# Helper
# =========================


def _to_str_id(value) -> str | None:
    """Reddit-ID immer als String speichern (kein Base36->Int-Cast)."""
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


def _to_ts_utc_naive(value) -> pd.Timestamp | None:
    """Nach UTC parsen und tz-naiv zurückgeben (kompatibel zu DuckDB TIMESTAMP)."""
    try:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return None
    if pd.isna(ts):
        return None
    # als naive UTC (ohne tz) zurückgeben – passt zu DuckDB TIMESTAMP
    return ts.tz_convert(None)


def _chunk(seq: list, n: int = 2000):
    """Yield in stabilen Batches (für große Inserts)."""
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


# =========================
# Main Functions
# =========================


def enrich_reddit_posts(
    con: duckdb.DuckDBPyConnection,
    posts_by_ticker: dict[str, list[dict]],
    _known_tickers: list[str] | None = None,
) -> int:
    """
    Enrich und persistiere Reddit-Posts in reddit_enriched.
    - sentiment_dict: Keyword-Heuristik
    - sentiment_weighted: sentiment_dict * ln(upvotes+1)
    Gibt die Anzahl der eingefügten/ersetzten Zeilen zurück.
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
            s_raw = analyze_sentiment(text)  # darf None sein
            sent = float(s_raw) if s_raw is not None else None
            weighted = None if sent is None else sent * math.log1p(upvotes)

            rows.append(
                {
                    "id": post_id,
                    "ticker": str(ticker).upper(),
                    "created_utc": created.to_pydatetime(),  # duckdb mag py datetime
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

    # Key-Paare für Upsert
    key_pairs: list[tuple[str, str]] = df[["id", "ticker"]].values.tolist()

    # Effizienter Batch-Delete via VALUES + USING Join (ein Roundtrip)
    con.execute(
        """
        DELETE FROM reddit_enriched
        USING (SELECT * FROM (VALUES {}) AS v(id, ticker)) AS del
        WHERE reddit_enriched.id = del.id AND reddit_enriched.ticker = del.ticker
        """.format(
            ", ".join(["(?, ?)"] * len(key_pairs))
        ),
        [x for pair in key_pairs for x in pair],
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
    values = df[cols].values.tolist()

    # Große Inserts sicher chunked
    inserted = 0
    stmt = (
        f"INSERT INTO reddit_enriched ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})"
    )
    for chunk in _chunk(values, 2000):
        con.executemany(stmt, chunk)
        inserted += len(chunk)

    return inserted


def compute_reddit_trends(con: duckdb.DuckDBPyConnection) -> int:
    """
    Aggregiere tägliche Reddit-Trends in reddit_trends.
    Voraussetzung: reddit_trends(date, ticker) hat PRIMARY KEY (date, ticker).
    Hinweis: rowcount kann in DuckDB 0 sein, obwohl geschrieben wurde.
    """
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
    return int(getattr(result, "rowcount", 0) or 0)


def compute_returns(
    con: duckdb.DuckDBPyConnection,
    horizon_days: tuple[int, ...] = (1, 3, 7),
) -> int:
    """
    Berechne Forward-Returns (1d/3d/7d) je Post in reddit_enriched.
    Wichtig: N = Anzahl HANDELStage (Index-Offset), nicht Kalendertage.
    """
    if not horizon_days:
        return 0

    df_posts = con.execute("SELECT id, ticker, created_utc FROM reddit_enriched").fetch_df()
    if df_posts.empty:
        return 0

    df_prices = con.execute("SELECT date, ticker, close FROM prices").fetch_df()
    if df_prices.empty:
        return 0

    df_prices["date"] = pd.to_datetime(df_prices["date"])
    df_posts["created_utc"] = pd.to_datetime(df_posts["created_utc"]).dt.normalize()

    updated = 0

    # Je Ticker effizient iterieren
    for tkr, grp in df_prices.groupby("ticker"):
        grp = grp.sort_values("date").reset_index(drop=True)
        posts_t = df_posts[df_posts["ticker"] == tkr]
        if posts_t.empty:
            continue

        closes = grp["close"].values

        for r in posts_t.itertuples(index=False):
            # Basis: erster Handelstag >= created_utc
            idx0 = grp["date"].searchsorted(r.created_utc, side="left")
            if idx0 >= len(grp):
                continue
            base_close = float(closes[idx0])

            ret_vals: list[float | None] = []
            for h in horizon_days:
                target = r.created_utc + pd.Timedelta(days=h)
                idxN = grp["date"].searchsorted(target, side="left")
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


def compute_reddit_sentiment(
    con: duckdb.DuckDBPyConnection,
    backfill_days: int = 7,
) -> tuple[int, int]:
    """Aggregate weighted sentiment per hour and per day.

    Results are written to ``reddit_sentiment_hourly`` and
    ``reddit_sentiment_daily`` using ``INSERT OR REPLACE``. Rows inside the
    backfill window are replaced. Returns a tuple of row counts for the recent
    hourly and daily windows.
    """

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS reddit_sentiment_hourly (
            created_utc TIMESTAMP,
            ticker TEXT,
            sentiment_dict DOUBLE,
            sentiment_weighted DOUBLE,
            posts INTEGER,
            PRIMARY KEY (created_utc, ticker)
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS reddit_sentiment_daily (
            date DATE,
            ticker TEXT,
            sentiment_dict DOUBLE,
            sentiment_weighted DOUBLE,
            posts INTEGER,
            PRIMARY KEY (date, ticker)
        )
        """
    )

    lookback = int(backfill_days or 0)
    lookback_hours = lookback * 24

    if lookback > 0:
        con.execute(
            f"DELETE FROM reddit_sentiment_hourly WHERE created_utc >= NOW() - INTERVAL {lookback_hours} HOUR",
        )
        con.execute(
            f"DELETE FROM reddit_sentiment_daily WHERE date >= CURRENT_DATE - INTERVAL {lookback} DAY",
        )
        hour_filter = f"AND created_utc >= NOW() - INTERVAL {lookback_hours} HOUR"
        day_filter = f"AND created_utc >= CURRENT_DATE - INTERVAL {lookback} DAY"
    else:
        con.execute("DELETE FROM reddit_sentiment_hourly")
        con.execute("DELETE FROM reddit_sentiment_daily")
        hour_filter = ""
        day_filter = ""

    con.execute(
        f"""
        INSERT OR REPLACE INTO reddit_sentiment_hourly
        SELECT
            DATE_TRUNC('hour', created_utc) AS created_utc,
            ticker,
            AVG(sentiment_dict) AS sentiment_dict,
            AVG(sentiment_weighted) AS sentiment_weighted,
            COUNT(*) AS posts
        FROM reddit_enriched
        WHERE sentiment_weighted IS NOT NULL {hour_filter}
        GROUP BY 1, 2
        """,
    )

    con.execute(
        f"""
        INSERT OR REPLACE INTO reddit_sentiment_daily
        SELECT
            DATE_TRUNC('day', created_utc) AS date,
            ticker,
            AVG(sentiment_dict) AS sentiment_dict,
            AVG(sentiment_weighted) AS sentiment_weighted,
            COUNT(*) AS posts
        FROM reddit_enriched
        WHERE sentiment_weighted IS NOT NULL {day_filter}
        GROUP BY 1, 2
        """,
    )

    rows_hourly = con.execute(
        "SELECT COUNT(*) FROM reddit_sentiment_hourly "
        + (f"WHERE created_utc >= NOW() - INTERVAL {lookback_hours} HOUR" if lookback > 0 else ""),
    ).fetchone()[0]
    rows_daily = con.execute(
        "SELECT COUNT(*) FROM reddit_sentiment_daily "
        + (f"WHERE date >= CURRENT_DATE - INTERVAL {lookback} DAY" if lookback > 0 else ""),
    ).fetchone()[0]

    return int(rows_hourly or 0), int(rows_daily or 0)

__all__ = [
    "enrich_reddit_posts",
    "compute_reddit_trends",
    "compute_returns",
    "compute_reddit_sentiment",
]
