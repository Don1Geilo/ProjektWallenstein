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
    "NVDA": ["nvidia"],
    "AMZN": ["amazon"],
    "AAPL": ["apple"],
    "MSFT": ["microsoft"],
    "GOOG": ["google", "alphabet"],
    "META": ["facebook", "meta"],
    "TSLA": ["tesla"],
}


def _load_aliases_from_file() -> None:
    """Merge additional ticker aliases from JSON/YAML file."""

    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / "data" / "ticker_aliases.json",
        root / "data" / "ticker_aliases.yaml",
        root / "data" / "ticker_aliases.yml",
    ]

    for path in candidates:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as fh:
                if path.suffix == ".json":
                    data = json.load(fh)
                elif path.suffix in {".yaml", ".yml"} and yaml:
                    data = yaml.safe_load(fh)
                else:
                    log.warning("Unsupported ticker alias file format: %s", path)
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
            log.warning("Could not load ticker aliases from %s: %s", path, exc)
        break  # use first existing file only


_load_aliases_from_file()

# ----------------------------
# Bestehende Funktion von dir
# ----------------------------
def fetch_reddit_posts(subreddit: str = "wallstreetbets", limit: int = 50) -> pd.DataFrame:
    """Return hot **and new** posts from ``subreddit`` as a ``DataFrame``.

    Additionally, the top comments for each post are fetched and returned as
    separate rows. Only interacts with the Reddit API; no database reads or
    writes occur here. Callers can persist the resulting frame if needed.
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
) -> Dict[str, List[dict]]:
    """Scrape, persist and organise Reddit posts.

    Posts from every subreddit are combined into a single DataFrame before new
    entries are merged into the ``reddit_posts`` table.  Afterwards, rows older
    than ``DATA_RETENTION_DAYS`` are purged to keep the table small.  Returns a
    mapping such as
    ``{"NVDA": [{"created_utc": <timestamp>, "text": "..."}, ...]}`` where each
    entry contains the post timestamp and text.
    """
    if not subreddits:
        subreddits = ["wallstreetbets", "wallstreetbetsGer", "mauerstrassenwetten"]

    # 1) neue Posts je Subreddit holen (Hot reicht als MVP; kann leicht auf 'new' umgestellt werden)
    frames = []
    for sub in subreddits:
        try:
            frames.append(fetch_reddit_posts(subreddit=sub, limit=limit_per_sub))
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
            con.execute(
                "INSERT INTO reddit_posts SELECT * FROM df_all ON CONFLICT(id) DO NOTHING"
            )
        log.info(f"Wrote {len(df_all)} posts to reddit_posts")

    # Alte Einträge entfernen
    purge_old_posts()

    # 2) Posts aus DB lesen
    df = _load_posts_from_db()

    # 3) Je Ticker Texte sammeln
    out: Dict[str, List[dict]] = {}
    for tkr in tickers:
        pats = _compile_patterns(tkr)
        bucket: List[dict] = []
        for _, row in df.iterrows():
            title = str(row.get("title", "") or "")
            text  = str(row.get("text", "") or "")
            if _post_matches_ticker(title, text, pats):
                # knapper Text-Chunk
                snippet = (title + "\n" + text).strip()
                if snippet:
                    # Längenlimit, damit analyze_sentiment nicht explodiert
                    bucket.append({"created_utc": row["created_utc"], "text": snippet[:2000]})
            # leichte Obergrenze pro Ticker (Performance)
            if len(bucket) >= 100:
                break
        log.debug(f"{tkr}: {len(bucket)} matched posts")
        out[tkr] = bucket
    return out
