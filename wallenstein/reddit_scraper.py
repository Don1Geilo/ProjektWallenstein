# wallenstein/reddit_scraper.py
from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from typing import Dict, List

import duckdb
import pandas as pd
import praw

from . import config

log = logging.getLogger("wallenstein.reddit")

# Gemeinsamer DB-Pfad (ENV erlaubt Override, sonst Default)
DB_PATH = os.getenv("WALLENSTEIN_DB_PATH", "wallenstein.duckdb")

# ----------------------------
# Bestehende Funktion von dir
# ----------------------------
def fetch_reddit_posts(subreddit: str = "wallstreetbets", limit: int = 50) -> pd.DataFrame:
    reddit = praw.Reddit(
        client_id=config.CLIENT_ID,
        client_secret=config.CLIENT_SECRET,
        user_agent=config.USER_AGENT
    )

    posts = []
    for post in reddit.subreddit(subreddit).hot(limit=limit):
        posts.append({
            "id": post.id,
            "title": post.title or "",
            "created_utc": datetime.fromtimestamp(post.created_utc, tz=timezone.utc),
            "text": post.selftext or ""
        })

    df = pd.DataFrame(posts)

    return df


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
    return patterns

def _post_matches_ticker(title: str, body: str, patterns: List[re.Pattern]) -> bool:
    t = title or ""
    b = body or ""
    for p in patterns:
        if p.search(t) or p.search(b):
            return True
    return False


# -------------------------------------------------------
# Öffentliche API: erwartet dein main.py
# -------------------------------------------------------
def update_reddit_data(tickers: List[str],
                       subreddits: List[str] | None = None,
                       limit_per_sub: int = 50) -> Dict[str, List[str]]:
    """
    Scraped neue Posts (pro Subreddit) -> schreibt DB -> liest DB -> mappt Texte je Ticker.
    Rückgabe: { "NVDA": ["titel + text", "..."], "AMZN": [...], ... }
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
            con.execute("""
                CREATE TABLE IF NOT EXISTS reddit_posts (
                    id VARCHAR,
                    title VARCHAR,
                    created_utc TIMESTAMP,
                    text VARCHAR
                )
            """)
            con.execute("DELETE FROM reddit_posts")
            con.register("df_all", df_all)
            con.execute("INSERT INTO reddit_posts SELECT * FROM df_all")
        log.info(f"Wrote {len(df_all)} posts to reddit_posts")

    # 2) Posts aus DB lesen
    df = _load_posts_from_db()

    # 3) Je Ticker Texte sammeln
    out: Dict[str, List[str]] = {}
    for tkr in tickers:
        pats = _compile_patterns(tkr)
        bucket: List[str] = []
        for _, row in df.iterrows():
            title = str(row.get("title", "") or "")
            text  = str(row.get("text", "") or "")
            if _post_matches_ticker(title, text, pats):
                # knapper Text-Chunk
                snippet = (title + "\n" + text).strip()
                if snippet:
                    # Längenlimit, damit analyze_sentiment nicht explodiert
                    bucket.append(snippet[:2000])
            # leichte Obergrenze pro Ticker (Performance)
            if len(bucket) >= 100:
                break
        out[tkr] = bucket
    return out
