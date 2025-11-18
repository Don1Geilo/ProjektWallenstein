from __future__ import annotations

import logging
import math
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache

import duckdb
import pandas as pd

from .aliases import add_alias, alias_map  # liefert Dict[str, Set[str]]
from .ticker_detection import SYMBOL_STOPWORDS, discover_new_tickers


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
    weekly_return: float | None = None


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



def _promote_unknown_candidates(
    con: duckdb.DuckDBPyConnection,
    candidates: list[TrendCandidate],
    known_symbols: set[str],
    df_recent: pd.DataFrame,
) -> set[str]:
    """Attempt to validate unknown symbols via automatic discovery.

    Returns the set of symbols that were promoted to *known* tickers.
    """

    unknown_map = {c.symbol: c for c in candidates if not c.is_known}
    if not unknown_map or "text" not in df_recent:
        return set()

    texts = df_recent["text"].dropna().astype(str).tolist()
    if not texts:
        return set()

    try:
        discovered = discover_new_tickers(texts, known=known_symbols)
    except RuntimeError as exc:  # pragma: no cover - optional dependency missing
        log.debug("Skipping auto discovery for unknown trends: %s", exc)
        return set()
    except Exception as exc:  # pragma: no cover - robustness
        log.warning("Automatic discovery of unknown trends failed: %s", exc)
        return set()

    if not discovered:
        return set()

    promoted: set[str] = set()
    for sym, meta in discovered.items():
        candidate = unknown_map.get(sym)
        if not candidate:
            continue
        candidate.is_known = True
        known_symbols.add(sym)
        promoted.add(sym)
        try:
            add_alias(con, sym, sym, source="auto-trend")
            for alias in sorted(meta.aliases):
                add_alias(con, sym, alias, source="auto-trend")
        except Exception as alias_exc:  # pragma: no cover - defensive
            log.debug("Persisting aliases for %s failed: %s", sym, alias_exc)

    if promoted:
        log.debug("Promoted unknown trend symbols: %s", ", ".join(sorted(promoted)))

    return promoted


# ---------- Kursentwicklung (7d) ----------
def _compute_weekly_return(df: pd.DataFrame) -> float | None:
    if df.empty or "close" not in df:
        return None

    tmp = df.copy()
    if "date" in tmp.columns:
        tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
    else:
        tmp = tmp.reset_index()
        tmp.rename(columns={tmp.columns[0]: "date"}, inplace=True)
        tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")

    tmp = tmp.dropna(subset=["date", "close"]).sort_values("date")
    if tmp.empty:
        return None

    latest_close = float(tmp["close"].iloc[-1])
    latest_date = tmp["date"].iloc[-1]
    if pd.isna(latest_date) or latest_close <= 0:
        return None

    cutoff = latest_date - pd.Timedelta(days=7)
    past = tmp[tmp["date"] <= cutoff]
    if past.empty:
        if len(tmp) < 2:
            return None
        reference = float(tmp["close"].iloc[0])
    else:
        reference = float(past["close"].iloc[-1])
    if reference <= 0:
        return None

    return float(latest_close / reference - 1.0)


def _weekly_return_from_db(
    con: duckdb.DuckDBPyConnection, symbol: str
) -> float | None:
    try:
        df = con.execute(
            """
            SELECT date, close
            FROM prices
            WHERE ticker = ?
            ORDER BY date DESC
            LIMIT 15
            """,
            [symbol],
        ).fetchdf()
    except duckdb.Error:
        return None

    if df.empty:
        return None

    return _compute_weekly_return(df)


@lru_cache(maxsize=64)
def _weekly_prices_from_yfinance(symbol: str) -> pd.DataFrame:
    if os.getenv("PYTEST_CURRENT_TEST"):
        return pd.DataFrame()
    try:
        import yfinance as yf  # type: ignore
    except Exception:
        return pd.DataFrame()

    try:
        hist = yf.Ticker(symbol).history(
            period="2mo", interval="1d", auto_adjust=False, actions=False
        )
    except Exception as exc:  # pragma: no cover - network best effort
        log.debug("yfinance history failed for %s: %s", symbol, exc)
        return pd.DataFrame()

    if hist is None or hist.empty or "Close" not in hist:
        return pd.DataFrame()

    hist = hist.dropna(subset=["Close"]).reset_index()
    if "Date" in hist.columns:
        hist = hist.rename(columns={"Date": "date", "Close": "close"})
    else:  # pragma: no cover - alternative index name
        hist = hist.rename(columns={hist.columns[0]: "date", "Close": "close"})

    return hist[["date", "close"]]


def _weekly_return_from_yfinance(symbol: str) -> float | None:
    df = _weekly_prices_from_yfinance(symbol)
    if df.empty:
        return None
    return _compute_weekly_return(df)



def fetch_weekly_returns(
    con: duckdb.DuckDBPyConnection,
    symbols: Iterable[str],
    max_symbols: int = 10,
) -> dict[str, float]:
    """Return up to ``max_symbols`` weekly returns for ``symbols``.

    Symbols are normalised and deduplicated before querying local prices or
    falling back to yfinance. Only successful lookups are returned.
    """

    results: dict[str, float] = {}
    for sym in symbols:
        if len(results) >= max_symbols:
            break
        symbol = _normalise_symbol(sym)
        if not symbol or symbol in results:
            continue
        val = _weekly_return_from_db(con, symbol)
        if val is None:
            val = _weekly_return_from_yfinance(symbol)
        if val is not None:
            results[symbol] = val
    return results


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


    if not candidates:
        return []

    promoted = _promote_unknown_candidates(con, candidates, known_symbols, df_win)
    if promoted:
        unknown_symbols.difference_update(promoted)



    if not candidates:
        return []


    sorted_candidates = sorted(
        candidates,
        key=lambda x: (int(x.is_known), x.trend, x.lift, x.mentions_24h),
        reverse=True,
    )

    fetch_order = [c.symbol for c in sorted_candidates]
    weekly_returns = fetch_weekly_returns(con, fetch_order, max_symbols=10)
    if weekly_returns:
        for cand in sorted_candidates:
            if cand.symbol in weekly_returns:
                cand.weekly_return = weekly_returns[cand.symbol]

    # Persistenz
    known_candidates = [c for c in sorted_candidates if c.is_known]
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

    return sorted_candidates


# ---------- Watchlist ----------
def auto_add_candidates_to_watchlist(
    con: duckdb.DuckDBPyConnection,
    notify_fn,
    max_new: int = 3,
    min_mentions: int = 30,
    min_lift: float = 4.0,
) -> list[str]:
    ensure_trending_tables(con)
    # Ensure the watchlist table exists – fresh databases on CI might not have
    # been initialised via the watchlist helpers yet.
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS watchlist (
            chat_id VARCHAR NOT NULL,
            symbol  VARCHAR NOT NULL,
            note    VARCHAR,
            UNIQUE(chat_id, symbol)
        )
        """
    )
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
                "INSERT OR REPLACE INTO watchlist (chat_id, symbol, note) VALUES ('_GLOBAL_', ?, ?)",
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

