# wallenstein/reddit_scraper.py
from __future__ import annotations

import itertools
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pandas as pd
import praw

from wallenstein.config import settings
from wallenstein.db_schema import ensure_tables, validate_df
from wallenstein.sentiment import analyze_sentiment_batch
from wallenstein.sentiment_analysis import analyze_sentiment

try:  # Optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover - not critical if missing
    yaml = None

try:  # Optional dependency for RSS scraping
    import feedparser  # type: ignore
except Exception:  # pragma: no cover - optional
    feedparser = None

log = logging.getLogger("wallenstein.reddit")
if settings.LOG_LEVEL.upper() == "DEBUG":
    log.setLevel(logging.DEBUG)

# Gemeinsamer DB-Pfad (ENV erlaubt Override, sonst Default)
DB_PATH = settings.WALLENSTEIN_DB_PATH

# Anzahl Tage, die Posts in der Datenbank behalten werden
DATA_RETENTION_DAYS = settings.DATA_RETENTION_DAYS

# Zu Tickersymbolen gehörende Firmennamen/Aliasse.  Die Zuordnung wird beim
# Import aus ``data/ticker_aliases.*`` geladen und ist hier zunächst leer.
TICKER_NAME_MAP: dict[str, list[str]] = {}


def _load_aliases_from_file(
    path: str | Path | None = None, aliases: dict[str, list[str]] | None = None
) -> None:
    """Merge additional ticker aliases from a dict or JSON/YAML file.

    Parameters
    ----------
    path:
        Optional file path pointing to a JSON or YAML file containing alias
        mappings. When provided the file is parsed on every call.
    aliases:
        Optional in-memory mapping ``{"TICKER": ["alias1", ...]}`` that is
        merged directly into :data:`TICKER_NAME_MAP`.
    """

    # 1) Merge explicit alias mapping first
    if aliases:
        for tkr, names in aliases.items():
            if not isinstance(names, list):
                continue
            bucket = TICKER_NAME_MAP.setdefault(tkr.upper(), [])
            for name in names:
                name = str(name).lower()
                if name not in bucket:
                    bucket.append(name)

    # 2) Determine candidate file paths
    root = Path(__file__).resolve().parents[1]
    candidates = (
        [Path(path)]
        if path
        else [
            root / "data" / "ticker_aliases.json",
            root / "data" / "ticker_aliases.yaml",
            root / "data" / "ticker_aliases.yml",
        ]
    )

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            with candidate.open("r", encoding="utf-8") as fh:
                if candidate.suffix == ".json":
                    data = json.load(fh)
                elif candidate.suffix in {".yaml", ".yml"} and yaml:
                    data = yaml.safe_load(fh)
                else:
                    log.warning("Unsupported ticker alias file format: %s", candidate)
                    data = {}

            for tkr, names in data.items():
                if not isinstance(names, list):
                    continue
                bucket = TICKER_NAME_MAP.setdefault(tkr.upper(), [])
                for name in names:
                    name = str(name).lower()
                    if name not in bucket:
                        bucket.append(name)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("Could not load ticker aliases from %s: %s", candidate, exc)
        break  # use first existing file only


_load_aliases_from_file()


# ----------------------------
# Bestehende Funktion von dir
# ----------------------------
def fetch_reddit_posts(
    subreddit: str = "wallstreetbets",
    limit: int = 50,
    include_comments: bool = False,
) -> pd.DataFrame:
    """Return hot and new posts from ``subreddit`` as a ``DataFrame``.

    Hot and new results are merged and deduplicated. If ``include_comments`` is
    ``True`` the top comments for each post are fetched and returned as separate
    rows. Only interacts with the Reddit API; no database reads or writes occur
    here. Callers can persist the resulting frame if needed.
    """

    reddit = praw.Reddit(
        client_id=settings.REDDIT_CLIENT_ID or os.getenv("CLIENT_ID"),
        client_secret=settings.REDDIT_CLIENT_SECRET or os.getenv("CLIENT_SECRET"),
        user_agent=settings.REDDIT_USER_AGENT or os.getenv("USER_AGENT") or "wallenstein",
    )

    posts = []
    sub = reddit.subreddit(subreddit)
    for post in itertools.chain(sub.hot(limit=limit), sub.new(limit=limit)):
        posts.append(
            {
                "id": post.id,
                "title": post.title or "",
                "created_utc": datetime.fromtimestamp(post.created_utc, tz=timezone.utc),
                "text": post.selftext or "",
                "upvotes": getattr(post, "score", 0),
            }
        )

        if include_comments:
            try:
                post.comments.replace_more(limit=0)
            except Exception:
                continue

            for comment in post.comments[:3]:
                posts.append(
                    {
                        "id": f"{post.id}_{comment.id}",
                        "title": "",
                        "created_utc": datetime.fromtimestamp(
                            getattr(comment, "created_utc", post.created_utc),
                            tz=timezone.utc,
                        ),
                        "text": comment.body or "",
                        "upvotes": getattr(comment, "score", 0),
                    }
                )

    df = pd.DataFrame(posts)
    df.drop_duplicates(subset="id", inplace=True)
    if not df.empty:
        combined = df["title"].fillna("") + " " + df["text"].fillna("")
        df["sentiment"] = combined.apply(analyze_sentiment)
    return df


def fetch_news_feed(url: str, limit: int = 50) -> pd.DataFrame:
    """Fetch items from an RSS/Atom feed as DataFrame.

    The function relies on :mod:`feedparser`.  Each entry is scored with the
    project's sentiment analyser and returned with an ``id``, ``title``,
    ``created_utc`` timestamp, ``text`` (summary) and ``sentiment`` score.
    """

    if not feedparser:  # pragma: no cover - optional dependency
        raise RuntimeError("feedparser package is required for news scraping")

    feed = feedparser.parse(url)
    rows: list[dict] = []
    for entry in getattr(feed, "entries", [])[:limit]:
        title = entry.get("title", "")
        summary = entry.get("summary", "")
        created = entry.get("published_parsed") or entry.get("updated_parsed")
        if created:
            from time import mktime

            created_dt = datetime.fromtimestamp(mktime(created), tz=timezone.utc)
        else:  # pragma: no cover - defensive
            created_dt = datetime.now(timezone.utc)
        text = f"{title} {summary}".strip()
        rows.append(
            {
                "id": entry.get("id") or entry.get("link") or title,
                "title": title,
                "created_utc": created_dt,
                "text": summary,
                "sentiment": analyze_sentiment(text),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------
# Hilfen: DB lesen & Ticker-Matching
# --------------------------------------
def _load_posts_from_db() -> pd.DataFrame:
    with duckdb.connect(DB_PATH) as con:
        try:
            df = con.execute(
                "SELECT id, created_utc, title, text, upvotes FROM reddit_posts ORDER BY created_utc DESC"
            ).fetch_df()
        except Exception:
            # Falls Tabelle noch nicht existiert
            df = pd.DataFrame(columns=["id", "created_utc", "title", "text", "upvotes"])
    return df


def _compile_patterns(ticker: str) -> list[re.Pattern]:
    """
    Erfasst Varianten wie NVDA, $NVDA, #NVDA, (NVDA).
    Vermeidet Treffer in Wörtern (z. B. 'ENVDA' soll nicht matchen).
    """
    # word boundary oder Sonderzeichen davor/danach
    safe = re.escape(ticker.upper())
    patterns = [
        re.compile(rf"(?<![A-Za-z0-9]){safe}(?![A-Za-z0-9])", re.IGNORECASE),  # NVDA
        re.compile(rf"[\$\#]\s*{safe}\b", re.IGNORECASE),  # $NVDA, #NVDA
        re.compile(rf"\(\s*{safe}\s*\)", re.IGNORECASE),  # (NVDA)
    ]

    for name in TICKER_NAME_MAP.get(ticker.upper(), []):
        # Sicherstellen, dass Namen als eigene Wörter erkannt werden
        safe_name = re.escape(name)
        patterns.append(re.compile(rf"(?<![A-Za-z0-9]){safe_name}(?![A-Za-z0-9])", re.IGNORECASE))

    return patterns


def purge_old_posts() -> None:
    """Remove posts older than ``DATA_RETENTION_DAYS`` from the database."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=DATA_RETENTION_DAYS)
    with duckdb.connect(DB_PATH) as con:
        try:
            con.execute(
                "DELETE FROM reddit_posts WHERE created_utc < ?",
                [cutoff],
            )
        except Exception:
            # Tabelle existiert noch nicht – dann gibt es nichts zu löschen
            pass


# -------------------------------------------------------
# Öffentliche API: erwartet dein main.py
# -------------------------------------------------------
def update_reddit_data(
    tickers: list[str],
    subreddits: list[str] | None = None,
    limit_per_sub: int = 50,
    include_comments: bool = False,
    aliases_path: str | Path | None = None,
    aliases: dict[str, list[str]] | None = None,
) -> dict[str, list[dict]]:
    """Scrape, persist and organise Reddit posts.

    Parameters
    ----------
    tickers:
        Symbols that should be matched in the collected posts.
    subreddits:
        Optional list of subreddits. Falls back to a small default set.
    limit_per_sub:
        Number of posts to fetch per subreddit.
    include_comments:
        Whether to also scan a few top-level comments.
    aliases_path / aliases:
        Extra ticker alias definitions.  ``aliases_path`` points to a JSON/YAML
        file that is reloaded on every call. ``aliases`` is an in-memory mapping
        in the same format.  Both are merged into
        :data:`TICKER_NAME_MAP` before processing.

    Returns
    -------
    dict
        Mapping such as ``{"NVDA": [{"created_utc": <timestamp>, "text": "..."},
        ...]}`` where each entry contains the post timestamp and text.
    """
    # Refresh alias data if provided
    _load_aliases_from_file(aliases_path, aliases)
    if not subreddits:
        subreddits = ["wallstreetbets", "wallstreetbetsGer", "mauerstrassenwetten"]

    # 1) neue Posts je Subreddit holen (Hot reicht als MVP; kann leicht auf 'new' umgestellt werden)
    frames: list[pd.DataFrame] = []

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                fetch_reddit_posts,
                subreddit=sub,
                limit=limit_per_sub,
                include_comments=include_comments,
            )
            for sub in subreddits
        ]
        for future in as_completed(futures):
            try:
                frames.append(future.result())
            except Exception:
                # Wenn ein Sub fehlschlägt, ignorieren – wir haben immer noch andere
                pass

    if frames:
        df_all = pd.concat(frames, ignore_index=True)
        if "upvotes" not in df_all.columns:
            df_all["upvotes"] = 0
        df_all = df_all.drop_duplicates(subset="id")

        if not df_all.empty:
            df_all = df_all[["id", "created_utc", "title", "text", "upvotes"]]
            ids = df_all["id"].tolist()
            with duckdb.connect(DB_PATH) as con:
                ensure_tables(con)
                before = con.execute("SELECT COUNT(*) FROM reddit_posts").fetchone()[0]
                existing_ids: set[str] = set()
                if ids:
                    placeholders = ",".join("?" for _ in ids)
                    query = f"SELECT id FROM reddit_posts WHERE id IN ({placeholders})"
                    existing_ids = set(con.execute(query, ids).fetch_df()["id"].tolist())
                    df_all = df_all[~df_all["id"].isin(existing_ids)]

                if not df_all.empty:
                    con.register("df_all", df_all)
                    validate_df(df_all, "reddit_posts")
                    con.execute(
                        "INSERT INTO reddit_posts (id, created_utc, title, text, upvotes) "
                        "SELECT id, created_utc, title, text, upvotes FROM df_all"
                    )
                after = con.execute("SELECT COUNT(*) FROM reddit_posts").fetchone()[0]
                added = max(0, after - before)
                log.info(f"Wrote {after} posts ({added} new)")

    # Alte Einträge entfernen
    purge_old_posts()

    # 2) Posts aus DB lesen
    df = _load_posts_from_db()
    if "upvotes" not in df.columns:
        df["upvotes"] = 0
    df["combined"] = df["title"].fillna("") + "\n" + df["text"].fillna("")

    # Nur Posts weiterverarbeiten, die überhaupt einen der Ticker erwähnen
    pattern_map = {
        t: "|".join(p.pattern for p in _compile_patterns(t)) for t in tickers
    }
    if pattern_map:
        combined_pattern = "|".join(pattern_map.values())
        df = df[
            df["combined"].str.contains(
                combined_pattern, regex=True, case=False, na=False
            )
        ]
        df = df.assign(
            **{
                t: df["combined"].str.contains(pat, regex=True, case=False, na=False)
                for t, pat in pattern_map.items()
            }
        )

    # 3) Je Ticker Texte sammeln
    out: dict[str, list[dict]] = {}
    for tkr in tickers:
        if tkr not in df.columns:
            continue
        bucket_df = (
            df.loc[df[tkr], ["id", "created_utc", "combined", "upvotes"]]
            .head(100)
            .rename(columns={"combined": "text"})
        )
        texts = bucket_df["text"].astype(str).str[:2000].tolist()
        bucket_df["text"] = texts
        bucket_df["sentiment"] = analyze_sentiment_batch(texts)
        bucket = bucket_df.to_dict(orient="records")
        log.debug(f"{tkr}: {len(bucket)} matched posts")
        out[tkr] = bucket
    return out


def detect_trending_tickers(
    ticker_posts: dict[str, list[dict]],
    window_hours: int = 24,
    baseline_days: int = 7,
    min_mentions: int = 3,
    ratio: float = 2.0,
) -> list[str]:
    """Identify tickers with unusually high mention counts.

    ``ticker_posts`` is the mapping returned by :func:`update_reddit_data`.
    A ticker is flagged as trending when the number of mentions in the last
    ``window_hours`` exceeds ``min_mentions`` and is ``ratio`` times higher than
    the preceding ``baseline_days``.
    """

    now = datetime.now(timezone.utc)
    window_start = now - timedelta(hours=window_hours)
    baseline_start = now - timedelta(days=baseline_days)
    trending: list[str] = []
    for tkr, posts in ticker_posts.items():
        recent = [p for p in posts if p.get("created_utc") and p["created_utc"] >= window_start]
        baseline = [
            p
            for p in posts
            if p.get("created_utc") and baseline_start <= p["created_utc"] < window_start
        ]
        if len(recent) >= min_mentions:
            base_count = len(baseline)
            if base_count == 0 or len(recent) / max(1, base_count) >= ratio:
                trending.append(tkr)
    return trending
