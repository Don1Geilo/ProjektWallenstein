from __future__ import annotations

import math
import re
from dataclasses import dataclass

import duckdb
import pandas as pd

from .aliases import alias_map  # liefert Dict[str, Set[str]]


# ---------- Datenklassen ----------
@dataclass
class TrendCandidate:
    symbol: str
    mentions_24h: int
    baseline_rate_per_h: float
    lift: float
    trend: float  # lift * log1p(mentions)


# ---------- Tabellen ----------
def ensure_trending_tables(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
    CREATE TABLE IF NOT EXISTS trending_candidates (
        symbol VARCHAR PRIMARY KEY,
        mentions_24h INTEGER,
        baseline_rate_per_h DOUBLE,
        lift DOUBLE,
        trend DOUBLE,
        first_seen TIMESTAMP,
        last_seen TIMESTAMP
    )"""
    )
    con.execute(
        """
    CREATE TABLE IF NOT EXISTS trending_candidates_history (
        ts TIMESTAMP,
        symbol VARCHAR,
        mentions_24h INTEGER,
        baseline_rate_per_h DOUBLE,
        lift DOUBLE,
        trend DOUBLE
    )"""
    )


# ---------- Regex-Matching vorbereiten ----------
def _compile_alias_patterns(amap: dict[str, set[str]]) -> dict[str, list[re.Pattern]]:
    """
    Erzeugt pro Symbol eine Liste Regex-Pattern:
      - Wortgrenzen für Plain-Words
      - Cashtag (#, $) Varianten
    """
    patmap: dict[str, list[re.Pattern]] = {}
    for sym, aliases in amap.items():
        pats: list[re.Pattern] = []
        seen: set[str] = set()
        for raw in aliases | {sym}:
            a = raw.strip()
            if not a or a.lower() in seen:
                continue
            seen.add(a.lower())

            # Cashtag-Varianten ($TSLA, #TSLA)
            pats.append(re.compile(rf"(?<!\w)[\$#]{re.escape(a)}(?!\w)", re.I))

            # Wortgrenzen für Namens-/Ticker-Strings
            # Beispiel: \btesla\b ; \btsla\b
            pats.append(re.compile(rf"\b{re.escape(a)}\b", re.I))
        patmap[sym] = pats
    return patmap


def _match_with_patterns(text: str, patmap: dict[str, list[re.Pattern]]) -> set[str]:
    if not text:
        return set()
    hits: set[str] = set()
    for sym, pats in patmap.items():
        for p in pats:
            if p.search(text):
                hits.add(sym)
                break
    # Konflikte bei Überschneidungen (optional: längster Alias gewinnt)
    return hits


# ---------- Zählen ----------
def _count_mentions(df: pd.DataFrame, patmap: dict[str, list[re.Pattern]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for txt in df["text"].astype(str):
        for s in _match_with_patterns(txt, patmap):
            counts[s] = counts.get(s, 0) + 1
    return counts


# Optional: gewichtete Zählung (Upvotes/Kommentare)
def _count_weighted_mentions(
    df: pd.DataFrame, patmap: dict[str, list[re.Pattern]]
) -> dict[str, float]:
    """Count mentions weighted by upvotes and comment volume."""
    counts: dict[str, float] = {}
    ups_series = df.get("ups", df.get("upvotes", 0))
    ups = pd.to_numeric(ups_series, errors="coerce").fillna(0).astype(float)
    com_series = df.get("num_comments", 0)
    com = pd.to_numeric(com_series, errors="coerce").fillna(0).astype(float)
    weights = 1.0 + ups.add(1).apply(math.log10) + 0.2 * com.add(1).apply(math.log10)
    texts = df["text"].astype(str)
    if len(texts) != len(weights):
        raise ValueError("Length mismatch between texts and weights")
    for txt, w in zip(texts, weights):  # noqa: B905
        for s in _match_with_patterns(txt, patmap):
            counts[s] = counts.get(s, 0.0) + float(w)
    return counts


# ---------- Hauptscan ----------
def scan_reddit_for_candidates(
    con: duckdb.DuckDBPyConnection,
    lookback_days: int = 7,
    window_hours: int = 24,
    min_mentions: int = 20,
    min_lift: float = 3.0,
    k_smooth: float = 0.5,  # Laplace-Glättung
    use_weighted: bool = False,  # Upvotes/Kommentare gewichten?
) -> list[TrendCandidate]:
    """
    Liefert sortierte Trendkandidaten (trend desc).
    Baseline = base_mentions / base_hours (geglättet). Lift = mentions24 / max(baseline*24, eps).
    trend = lift * log1p(mentions24)
    """
    ensure_trending_tables(con)
    amap = alias_map(con, include_ticker_itself=True)
    if not amap:
        return []

    patmap = _compile_alias_patterns(amap)

    # Fenster laden (UTC im DB; hier neutral belassen)
    cols = {row[1] for row in con.execute("PRAGMA table_info('reddit_posts')").fetchall()}
    num_comments_expr = "num_comments" if "num_comments" in cols else "0 AS num_comments"
    df_win = con.execute(
        f"""
        SELECT created_utc, text, upvotes AS ups, {num_comments_expr}
        FROM reddit_posts
        WHERE created_utc >= NOW() - INTERVAL {int(window_hours)} HOUR
    """
    ).fetchdf()

    if df_win.empty:
        return []

    df_base = con.execute(
        f"""
        SELECT created_utc, text, upvotes AS ups, {num_comments_expr}
        FROM reddit_posts
        WHERE created_utc >= NOW() - INTERVAL {int(lookback_days*24)} HOUR
          AND created_utc <  NOW() - INTERVAL {int(window_hours)} HOUR
    """
    ).fetchdf()

    # Zählen
    win_counts_raw = (
        _count_weighted_mentions(df_win, patmap)
        if use_weighted
        else _count_mentions(df_win, patmap)
    )
    base_counts_raw = (
        _count_weighted_mentions(df_base, patmap)
        if (use_weighted and not df_base.empty)
        else (_count_mentions(df_base, patmap) if not df_base.empty else {})
    )

    # Stunden
    base_hours = max(1.0, float(lookback_days * 24 - window_hours))
    # Baseline-Rate/h + Glättung
    baseline_rate_per_h: dict[str, float] = {}
    for s in amap.keys():
        base_val = float(base_counts_raw.get(s, 0.0))
        # Laplace: +k in Zähler und Nenner
        baseline_rate_per_h[s] = (base_val + k_smooth) / (base_hours + k_smooth)

    # Kandidaten bauen
    candidates: list[TrendCandidate] = []
    for s, cnt in win_counts_raw.items():
        mentions = int(round(cnt)) if use_weighted else int(cnt)
        if mentions < min_mentions:
            continue
        rate = baseline_rate_per_h.get(s, k_smooth / (base_hours + k_smooth))
        # Erwartung für 24h:
        expected_24h = max(1e-6, rate * window_hours)
        lift = float(mentions) / expected_24h
        trend = lift * math.log1p(max(0, mentions))
        if lift >= min_lift:
            candidates.append(TrendCandidate(s, mentions, rate, lift, trend))

    # Persistenz
    now = con.execute("SELECT NOW()").fetchone()[0]
    if candidates:
        data = [
            (now, c.symbol, c.mentions_24h, c.baseline_rate_per_h, c.lift, c.trend)
            for c in candidates
        ]
        con.executemany(
            """
            INSERT INTO trending_candidates_history (ts, symbol, mentions_24h, baseline_rate_per_h, lift, trend)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            data,
        )

        # "latest" Tabelle aktualisieren (DuckDB: MERGE verfügbar; sonst DELETE+INSERT)
        try:
            con.register(
                "tmp_trends",
                pd.DataFrame(
                    [
                        (c.symbol, c.mentions_24h, c.baseline_rate_per_h, c.lift, c.trend)
                        for c in candidates
                    ],
                    columns=["symbol", "mentions_24h", "baseline_rate_per_h", "lift", "trend"],
                ),
            )
            con.execute(
                """
                MERGE INTO trending_candidates t
                USING tmp_trends s
                ON t.symbol = s.symbol
                WHEN MATCHED THEN UPDATE SET
                    mentions_24h = s.mentions_24h,
                    baseline_rate_per_h = s.baseline_rate_per_h,
                    lift = s.lift,
                    trend = s.trend,
                    last_seen = NOW()
                WHEN NOT MATCHED THEN INSERT
                    (symbol, mentions_24h, baseline_rate_per_h, lift, trend, first_seen, last_seen)
                VALUES (s.symbol, s.mentions_24h, s.baseline_rate_per_h, s.lift, s.trend, NOW(), NOW())
            """
            )
        except Exception:
            # Fallback: DELETE+INSERT (kompatibel mit älteren DuckDBs)
            symbols = [c.symbol for c in candidates]
            con.executemany(
                "DELETE FROM trending_candidates WHERE symbol = ?", [(s,) for s in symbols]
            )
            con.executemany(
                """
                INSERT INTO trending_candidates(symbol, mentions_24h, baseline_rate_per_h, lift, trend, first_seen, last_seen)
                VALUES (?, ?, ?, ?, ?, NOW(), NOW())
            """,
                [
                    (c.symbol, c.mentions_24h, c.baseline_rate_per_h, c.lift, c.trend)
                    for c in candidates
                ],
            )

    # Sortierung: erst trend, dann lift, dann mentions
    return sorted(candidates, key=lambda x: (x.trend, x.lift, x.mentions_24h), reverse=True)


# ---------- Watchlist ----------
def auto_add_candidates_to_watchlist(
    con: duckdb.DuckDBPyConnection,
    notify_fn,
    max_new: int = 3,
    min_mentions: int = 30,
    min_lift: float = 4.0,
) -> list[str]:
    ensure_trending_tables(con)
    rows = con.execute(
        """
        SELECT symbol, mentions_24h, baseline_rate_per_h, lift, trend
        FROM trending_candidates
        ORDER BY trend DESC, lift DESC, mentions_24h DESC
        LIMIT 20
    """
    ).fetchall()
    if not rows:
        return []

    wl = {s for (s,) in con.execute("SELECT DISTINCT symbol FROM watchlist").fetchall()}
    added: list[str] = []
    for sym, mentions, _base_rate, lift, trend in rows:
        if len(added) >= max_new:
            break
        if sym in wl:
            continue
        if mentions is None or lift is None:
            continue
        if mentions >= min_mentions and lift >= min_lift:
            con.execute(
                "INSERT OR REPLACE INTO watchlist (chat_id, symbol, note) VALUES ('GLOBAL', ?, ?)",
                [sym, f"auto-added {mentions} m24h, lift {lift:.1f}, trend {trend:.2f}"],
            )
            added.append(sym)

    if added and notify_fn:
        try:
            notify_fn("⚡ Neue Trend-Ticker: " + ", ".join(added))
        except Exception:
            pass
    return added
