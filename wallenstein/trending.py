from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass

import duckdb
import pandas as pd

from .aliases import alias_map  # liefert Dict[str, Set[str]]
from .ticker_detection import SYMBOL_STOPWORDS


log = logging.getLogger(__name__)

CASHTAG_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])[#$]\s*([A-Za-z][A-Za-z0-9.\-]{0,9})(?![A-Za-z0-9])",
    re.IGNORECASE,
)


# ---------- Datenklassen ----------
@dataclass
class TrendCandidate:
    symbol: str
    mentions_24h: int
    baseline_rate_per_h: float
    lift: float
    trend: float  # lift * log1p(mentions)
    is_known: bool = False


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
    # Ergänze um frei erkannte Cashtags
    for raw in CASHTAG_PATTERN.findall(text):
        symbol = _normalise_symbol(raw)
        if symbol:
            hits.add(symbol)
    # Konflikte bei Überschneidungen (optional: längster Alias gewinnt)
    return hits


# ---------- Hilfsfunktionen ----------
def _fetch_distinct_symbols(
    con: duckdb.DuckDBPyConnection, table: str, column: str
) -> set[str]:
    """Return distinct, normalised symbols from ``table.column``."""

    try:
        rows = con.execute(
            f"SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL"
        ).fetchall()
    except duckdb.Error:
        return set()

    symbols: set[str] = set()
    for (value,) in rows:
        symbol = _normalise_symbol(value)
        if symbol:
            symbols.add(symbol)
    return symbols


# ---------- Cashtag-Extraktion ----------
def _extract_cashtags(text: str | None) -> set[str]:
    if not text:
        return set()
    results: set[str] = set()
    for match in CASHTAG_PATTERN.finditer(str(text)):
        symbol = _normalise_symbol(match.group(1))
        if symbol:
            results.add(symbol)
    return results


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
    # include basic aliases plus WordNet synonyms for broader matching

    base_aliases = alias_map(con, include_ticker_itself=True, use_synonyms=True)
    amap: dict[str, set[str]] = {sym: set(names) for sym, names in base_aliases.items()}
    known_symbols: set[str] = set(amap.keys())

    # Weitere Quellen: Watchlist, Preise, historische Trends
    for source_table, column in (
        ("watchlist", "symbol"),
        ("prices", "ticker"),
        ("reddit_enriched", "ticker"),
        ("reddit_trends", "ticker"),
    ):
        for sym in _fetch_distinct_symbols(con, source_table, column):
            known_symbols.add(sym)
            amap.setdefault(sym, set()).add(sym)

    patmap = _compile_alias_patterns(amap) if amap else {}


    # Fenster laden (UTC im DB; hier neutral belassen)
    cols = {row[1] for row in con.execute("PRAGMA table_info('reddit_posts')").fetchall()}
    num_comments_expr = "num_comments" if "num_comments" in cols else "0 AS num_comments"
    text_expr = "COALESCE(title, '') || ' ' || COALESCE(text, '') AS text"
    window_hours_int = int(window_hours)
    lookback_hours = int(lookback_days * 24)
    df_win = con.execute(
        f"""
        SELECT created_utc, {text_expr}, upvotes AS ups, {num_comments_expr}
        FROM reddit_posts
        WHERE created_utc >= NOW() - INTERVAL {window_hours_int} HOUR
    """
    ).fetchdf()

    if df_win.empty:
        return []

    df_base = con.execute(
        f"""
        SELECT created_utc, {text_expr}, upvotes AS ups, {num_comments_expr}
        FROM reddit_posts
        WHERE created_utc >= NOW() - INTERVAL {lookback_hours} HOUR
          AND created_utc <  NOW() - INTERVAL {window_hours_int} HOUR
    """
    ).fetchdf()

    # Neue Cashtags aus den Texten extrahieren, um das Alias-Mapping zu erweitern
    for df in (df_win, df_base):
        if df.empty or "text" not in df:
            continue
        for text in df["text"].astype(str):
            for tag in _extract_cashtags(text):
                amap.setdefault(tag, set()).add(tag)

    if not amap:
        return []

    patmap = _compile_alias_patterns(amap)

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
    unknown_symbols: set[str] = set()
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
            is_known = s in known_symbols
            if not is_known:
                unknown_symbols.add(s)
            candidates.append(TrendCandidate(s, mentions, rate, lift, trend, is_known=is_known))

    # Persistenz
    known_candidates = [c for c in candidates if c.is_known]
    if unknown_symbols:
        log.debug("Ungeprüfte Trend-Symbole ignoriert: %s", sorted(unknown_symbols))

    if known_candidates:
        now = con.execute("SELECT NOW()").fetchone()[0]
        data = [
            (now, c.symbol, c.mentions_24h, c.baseline_rate_per_h, c.lift, c.trend)
            for c in known_candidates
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
                        for c in known_candidates
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
            symbols = [c.symbol for c in known_candidates]
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
                    for c in known_candidates
                ],
            )

    # Sortierung: bekannte Symbole zuerst, dann trend, lift, mentions
    return sorted(
        candidates,
        key=lambda x: (int(x.is_known), x.trend, x.lift, x.mentions_24h),
        reverse=True,
    )


# ---------- Watchlist ----------
def auto_add_candidates_to_watchlist(
    con: duckdb.DuckDBPyConnection,
    notify_fn,
    max_new: int = 3,
    min_mentions: int = 30,
    min_lift: float = 4.0,
) -> list[str]:
    ensure_trending_tables(con)
    known_symbols: set[str] = set(
        alias_map(con, include_ticker_itself=True, use_synonyms=False).keys()
    )
    for sym in _fetch_distinct_symbols(con, "watchlist", "symbol"):
        known_symbols.add(sym)
    for sym in _fetch_distinct_symbols(con, "prices", "ticker"):
        known_symbols.add(sym)
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
        if sym not in known_symbols:
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


def _normalise_symbol(sym: str | None) -> str | None:
    symbol = str(sym or "").strip().upper()
    if not symbol:
        return None
    symbol = symbol.strip("$# \t\r\n.,:;!?")
    if not symbol:
        return None
    symbol = re.sub(r"\s+", "", symbol)
    if len(symbol) > 12:
        return None
    if not any(ch.isalpha() for ch in symbol):
        return None
    if symbol in SYMBOL_STOPWORDS:
        return None
    if not re.fullmatch(r"[A-Z0-9]+(?:[.\-][A-Z0-9]+)*", symbol):
        return None
    return symbol

