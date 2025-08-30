from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

import duckdb
import pandas as pd

from .aliases import alias_map


@dataclass
class TrendCandidate:
    symbol: str
    mentions_24h: int
    baseline_daily: float
    lift: float

def ensure_trending_tables(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
    CREATE TABLE IF NOT EXISTS trending_candidates (
        symbol VARCHAR PRIMARY KEY,
        mentions_24h INTEGER,
        baseline_daily DOUBLE,
        lift DOUBLE,
        first_seen TIMESTAMP,
        last_seen TIMESTAMP
    )
    """)

def _match_with_aliases(text: str, amap: Dict[str, Set[str]]) -> Set[str]:
    if not text:
        return set()
    u = text.upper()
    hits: Set[str] = set()
    for tkr, aliases in amap.items():
        for a in aliases:
            if a and a.upper() in u:
                hits.add(tkr)
                break
    return hits

def scan_reddit_for_candidates(
    con: duckdb.DuckDBPyConnection,
    lookback_days: int = 7,
    window_hours: int = 24,
    min_mentions: int = 20,
    min_lift: float = 3.0,
) -> List[TrendCandidate]:
    ensure_trending_tables(con)
    amap = alias_map(con, include_ticker_itself=True)
    if not amap:
        return []

    df_win = con.execute(f"""
        SELECT created_utc, text
        FROM reddit_posts
        WHERE created_utc >= NOW() - INTERVAL {int(window_hours)} HOUR
    """).fetchdf()
    if df_win.empty:
        return []

    df_base = con.execute(f"""
        SELECT created_utc, text
        FROM reddit_posts
        WHERE created_utc >= DATE_TRUNC('day', NOW()) - INTERVAL {int(lookback_days)} DAY
          AND created_utc < NOW() - INTERVAL {int(window_hours)} HOUR
    """).fetchdf()

    def count_mentions(df: pd.DataFrame) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for txt in df["text"].astype(str):
            for s in _match_with_aliases(txt, amap):
                counts[s] = counts.get(s, 0) + 1
        return counts

    win_counts = count_mentions(df_win)
    base_counts = count_mentions(df_base) if not df_base.empty else {}
    days = max(1, lookback_days - (window_hours // 24))
    baseline_daily = {s: (base_counts.get(s, 0) / days) for s in amap.keys()}

    candidates: List[TrendCandidate] = []
    for s, cnt in win_counts.items():
        base = baseline_daily.get(s, 0.0)
        lift = (cnt / base) if base > 0 else float("inf")
        if cnt >= min_mentions and (lift >= min_lift or base == 0):
            candidates.append(TrendCandidate(s, cnt, base, lift))

    if candidates:
        rows = [(c.symbol, c.mentions_24h, c.baseline_daily, c.lift) for c in candidates]
        con.executemany("""
            INSERT INTO trending_candidates(symbol, mentions_24h, baseline_daily, lift, first_seen, last_seen)
            VALUES (?, ?, ?, ?, NOW(), NOW())
            ON CONFLICT(symbol) DO UPDATE SET
                mentions_24h = EXCLUDED.mentions_24h,
                baseline_daily = EXCLUDED.baseline_daily,
                lift = EXCLUDED.lift,
                last_seen = NOW()
        """, rows)

    return sorted(candidates, key=lambda x: (x.lift, x.mentions_24h), reverse=True)

def auto_add_candidates_to_watchlist(
    con: duckdb.DuckDBPyConnection,
    notify_fn,
    max_new: int = 3,
    min_mentions: int = 30,
    min_lift: float = 4.0,
) -> List[str]:
    ensure_trending_tables(con)
    rows = con.execute("""
        SELECT symbol, mentions_24h, baseline_daily, lift
        FROM trending_candidates
        ORDER BY lift DESC, mentions_24h DESC
        LIMIT 20
    """).fetchall()
    if not rows:
        return []

    wl = {s for (s,) in con.execute("SELECT DISTINCT symbol FROM watchlist").fetchall()}
    added: List[str] = []
    for sym, mentions, _base, lift in rows:
        if len(added) >= max_new:
            break
        if sym in wl:
            continue
        if mentions is None or lift is None:
            continue
        if mentions >= min_mentions and lift >= min_lift:
            con.execute(
                "INSERT OR REPLACE INTO watchlist (chat_id, symbol, note) VALUES ('GLOBAL', ?, ?)",
                [sym, f"auto-added {mentions} m24h, lift {lift:.1f}"],
            )
            added.append(sym)

    if added and notify_fn:
        try:
            notify_fn("⚡ Neue Trend-Ticker: " + ", ".join(added))
        except Exception:
            pass
    return added