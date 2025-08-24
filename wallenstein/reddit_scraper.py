# wallenstein/reddit_scraper.py
from __future__ import annotations

import logging
import os
import re
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

import duckdb
import pandas as pd
import praw

try:  # Optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover - not critical if missing
    yaml = None

from . import config

log = logging.getLogger("wallenstein.reddit")
if os.getenv("WALLENSTEIN_LOG_LEVEL", "").upper() == "DEBUG":
    log.setLevel(logging.DEBUG)

# Gemeinsamer DB-Pfad (ENV erlaubt Override, sonst Default)
DB_PATH = os.getenv("WALLENSTEIN_DB_PATH", "wallenstein.duckdb")

# Anzahl Tage, die Posts in der Datenbank behalten werden
DATA_RETENTION_DAYS = int(os.getenv("DATA_RETENTION_DAYS", "30"))

# Bekannte Firmennamen zu Tickersymbolen.  Diese Liste ist keineswegs
# vollständig, deckt aber einige der üblichen Verdächtigen ab.  Für jeden
# Ticker können mehrere Varianten des Firmennamens angegeben werden, die im
# Text erkannt werden sollen.
TICKER_NAME_MAP: Dict[str, List[str]] = {
    "NVDA": ["nvidia", "nividia", "nvidea"],
    "AMZN": ["amazon", "amzon", "amazn"],
    "AAPL": ["apple", "aple", "appl"],
    "MSFT": ["microsoft", "micosoft", "micro soft"],
    "GOOG": ["google", "alphabet", "googel", "gogle"],
    "META": ["facebook", "meta", "metta", "facebok"],
    "TSLA": ["tesla", "tesler", "tesal"],
    "RHM": ["rheinmetall", "rheiner"],
}


def _load_aliases_from_file(
    path: str | Path | None = None, aliases: Dict[str, List[str]] | None = None
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
    candidates = [Path(path)] if path else [
        root / "data" / "ticker_aliases.json",
        root / "data" / "ticker_aliases.yaml",
        root / "data" / "ticker_aliases.yml",
    ]

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
                    log.warning(
                        "Unsupported ticker alias file format: %s", candidate
                    )
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
            log.warning(
                "Could not load ticker aliases from %s: %s", candidate, exc
            )
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
    """Return hot **and new** posts from ``subreddit`` as a ``DataFrame``.

    If ``include_comments`` is ``True`` the top comments for each post are
    fetched and returned as separate rows. Only interacts with the Reddit API;
    no database reads or writes occur here. Callers can persist the resulting
    frame if needed.
    """

    reddit = praw.Reddit(
        client_id=config.CLIENT_ID,
        client_secret=config.CLIENT_SECRET,
        user_agent=config.USER_AGENT,
    )

    posts = []
    sub = reddit.subreddit(subreddit)
    for post in itertools.chain(sub.hot(limit=limit), sub.new(limit=limit)):
        posts.append({
            "id": post.id,
            "title": post.title or "",
            "created_utc": datetime.fromtimestamp(post.created_utc, tz=timezone.utc),
            "text": post.selftext or "",
        })

        if include_comments:
            try:
                post.comments.replace_more(limit=0)
            except Exception:
                continue

            for comment in post.comments[:3]:
                posts.append({
                    "id": f"{post.id}_{comment.id}",
                    "title": "",
                    "created_utc": datetime.fromtimestamp(
                        getattr(comment, "created_utc", post.created_utc),
                        tz=timezone.utc,
                    ),
                    "text": comment.body or "",
                })

    return pd.DataFrame(posts)


# --------------------------------------
# Hilfen: DB lesen & Ticker-Matching
# --------------------------------------
def _load_posts_from_db() -> pd.DataFrame:
    with duckdb.connect(DB_PATH) as con:
        try:
            df = con.execute("SELECT * FROM reddit_posts ORDER BY created_utc DESC").fetch_df()
        except Exception:
            # Falls Tabelle noch nicht existiert
            df = pd.DataFrame(columns=["id", "title", "created_utc", "text"])
    return df

def _compile_patterns(ticker: str) -> List[re.Pattern]:
    """
    Erfasst Varianten wie NVDA, $NVDA, #NVDA, (NVDA).
    Vermeidet Treffer in Wörtern (z. B. 'ENVDA' soll nicht matchen).
    """
    # word boundary oder Sonderzeichen davor/danach
    safe = re.escape(ticker.upper())
    patterns = [
        re.compile(rf"(?<![A-Za-z0-9]){safe}(?![A-Za-z0-9])", re.IGNORECASE),  # NVDA
        re.compile(rf"[\$\#]\s*{safe}\b", re.IGNORECASE),                      # $NVDA, #NVDA
        re.compile(rf"\(\s*{safe}\s*\)", re.IGNORECASE),                       # (NVDA)
    ]

    for name in TICKER_NAME_MAP.get(ticker.upper(), []):
        # Sicherstellen, dass Namen als eigene Wörter erkannt werden
        safe_name = re.escape(name)
        patterns.append(
            re.compile(rf"(?<![A-Za-z0-9]){safe_name}(?![A-Za-z0-9])", re.IGNORECASE)
        )

    return patterns

def _post_matches_ticker(title: str, body: str, patterns: List[re.Pattern]) -> bool:
    t = title or ""
    b = body or ""
    for p in patterns:
        if p.search(t) or p.search(b):
            return True
    return False


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
    tickers: List[str],
    subreddits: List[str] | None = None,
    limit_per_sub: int = 50,
    include_comments: bool = False,
    aliases_path: str | Path | None = None,
    aliases: Dict[str, List[str]] | None = None,
) -> Dict[str, List[dict]]:
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
    frames = []

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(fetch_reddit_posts, subreddit=sub, limit=limit_per_sub)
            for sub in subreddits
        ]
        for future in as_completed(futures):
            try:
                frames.append(future.result())
            except Exception:
                # Wenn ein Sub fehlschlägt, ignorieren – wir haben immer noch andere
                pass

    for sub in subreddits:
        try:
            frames.append(
                fetch_reddit_posts(
                    subreddit=sub,
                    limit=limit_per_sub,
                    include_comments=include_comments,
                )
            )
        except Exception:
            # Wenn ein Sub fehlschlägt, ignorieren – wir haben immer noch andere
            pass


    if frames:
        df_all = pd.concat(frames, ignore_index=True)
        df_all.drop_duplicates(subset="id", inplace=True)
        with duckdb.connect(DB_PATH) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS reddit_posts (
                    id VARCHAR,
                    title VARCHAR,
                    created_utc TIMESTAMP,
                    text VARCHAR
                )
                """
            )
            # Ensure uniqueness on id for existing tables
            con.execute(
                """
                DELETE FROM reddit_posts
                WHERE rowid NOT IN (
                    SELECT MIN(rowid) FROM reddit_posts GROUP BY id
                )
                """
            )
            con.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS reddit_posts_id_idx ON reddit_posts(id)"
            )
            con.register("df_all", df_all)
            cur = con.execute(
                "INSERT INTO reddit_posts SELECT * FROM df_all ON CONFLICT(id) DO NOTHING"
            )
        log.info(
            f"Wrote {len(df_all)} posts to reddit_posts ({cur.rowcount} new)"
        )

    # Alte Einträge entfernen
    purge_old_posts()

    # 2) Posts aus DB lesen
    df = _load_posts_from_db()
    df["combined"] = df["title"].fillna("") + "\n" + df["text"].fillna("")

    # 3) Je Ticker Texte sammeln
    out: Dict[str, List[dict]] = {}
    for tkr in tickers:
        pattern = "|".join(p.pattern for p in _compile_patterns(tkr))
        mask = df["combined"].str.contains(pattern, regex=True, case=False, na=False)
        bucket_df = (
            df.loc[mask, ["created_utc", "combined"]]
            .head(100)
            .rename(columns={"combined": "text"})
        )
        bucket_df["text"] = bucket_df["text"].str[:2000]
        bucket = bucket_df.to_dict(orient="records")
        log.debug(f"{tkr}: {len(bucket)} matched posts")
        out[tkr] = bucket
    return out
