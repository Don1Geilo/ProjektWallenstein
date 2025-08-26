import io
import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import duckdb
import pandas as pd
import requests
import yfinance as yf
from requests.adapters import HTTPAdapter
from requests.exceptions import HTTPError
from urllib3.util.retry import Retry

from wallenstein.config import settings
from wallenstein.db_schema import ensure_tables, validate_df

# ---- Configuration ----
MAX_RETRIES = 6
CHUNK_SIZE = 20
# limit yahoo concurrency to reduce risk of 429s
YAHOO_MAX_WORKERS = 3
# User-Agent for Stooq requests (some environments return 403 for default UA)
STOOQ_HEADERS = {"User-Agent": settings.STOOQ_USER_AGENT}

# Datenquelle: yahoo (default) oder stooq (mit Yahoo-Fallback)
# 'hybrid' bleibt aus Kompatibilitätsgründen als Alias zu stooq bestehen
DATA_SOURCE = settings.WALLENSTEIN_DATA_SOURCE

log = logging.getLogger(__name__)


# ---- DuckDB Helpers ----
def _connect(db_path: str) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(db_path)


def _ensure_prices_table(con: duckdb.DuckDBPyConnection):
    # Typ des Objekts "prices" ermitteln: None | 'BASE TABLE' | 'VIEW'
    row = con.execute(
        """
        SELECT table_type
        FROM information_schema.tables
        WHERE lower(table_name) = 'prices'
        LIMIT 1
    """
    ).fetchone()
    obj_type = row[0] if row else None

    # Nur wenn es wirklich eine VIEW ist, droppen
    if obj_type == "VIEW":
        con.execute("DROP VIEW prices")

    # Tabelle sicherstellen (no-op, falls schon vorhanden)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS prices (
            date DATE,
            ticker VARCHAR,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            adj_close DOUBLE,
            volume BIGINT
        )
    """
    )
    # Index für häufige Abfragen (Ticker + Datum)
    try:
        con.execute("CREATE INDEX IF NOT EXISTS prices_ticker_date_idx ON prices(ticker, date)")
    except Exception:
        pass  # pragma: no cover - defensive: ältere DuckDB-Versionen ohne Index-Unterstützung


def _latest_dates_per_ticker(
    con: duckdb.DuckDBPyConnection, tickers: list[str]
) -> dict[str, date | None]:
    if not tickers:
        return {}
    q = """
        SELECT ticker, max(date) AS last_date
        FROM prices
        WHERE ticker IN ({})
        GROUP BY ticker
    """.format(
        ", ".join("'" + t + "'" for t in tickers)
    )
    try:
        df = con.execute(q).fetchdf()
        # last_date kommt als datetime.date (DuckDB DATE)
        return {row["ticker"]: row["last_date"] for _, row in df.iterrows()}
    except Exception:
        return {}


def _retry_sleep(attempt: int):
    """Sleep with exponential backoff and jitter."""
    base = 1.0 * (2**attempt)
    time.sleep(base + random.random())


# ---- Stooq (stable, CSV) ----
def _stooq_symbol(t: str) -> str:
    # Stooq erwartet z.B. nvda.us, amzn.us, smci.us
    return f"{t.lower()}.us"


def _stooq_fetch_one(
    ticker: str, start: pd.Timestamp | None = None, session: requests.Session | None = None
) -> pd.DataFrame:
    """Fetch a single ticker from Stooq with basic retry logic."""
    sym = _stooq_symbol(ticker)
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    sess = session or requests
    for attempt in range(MAX_RETRIES):
        try:
            r = sess.get(url, timeout=20, headers=STOOQ_HEADERS)
            if not r.ok or not r.text:
                raise RuntimeError("empty response")
            df = pd.read_csv(io.StringIO(r.text))
            if df.empty:
                raise RuntimeError("empty dataframe")
            # Stooq Spalten: Date,Open,High,Low,Close,Volume
            df = df.rename(
                columns={
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )
            df["ticker"] = ticker
            df["date"] = pd.to_datetime(df["date"]).dt.date
            if start is not None:
                start_d = pd.to_datetime(start).date()
                df = df[df["date"] >= start_d]
            # adj_close nicht vorhanden
            df["adj_close"] = pd.NA
            return df[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]]
        except Exception:
            _retry_sleep(attempt)
    return pd.DataFrame()


def _stooq_fetch_many(
    tickers: list[str], start_map: dict[str, pd.Timestamp] | None = None
) -> pd.DataFrame:
    """Fetch multiple tickers from Stooq concurrently."""
    if not tickers:
        return pd.DataFrame(
            columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
        )

    results: list[pd.DataFrame] = []

    def _fetch(t: str) -> pd.DataFrame:
        start = start_map.get(t) if start_map else None
        return _stooq_fetch_one(t, start=start)

    with ThreadPoolExecutor(max_workers=min(5, len(tickers))) as ex:
        futures = {ex.submit(_fetch, t): t for t in tickers}
        for fut in as_completed(futures):
            df = fut.result()
            if not df.empty:
                results.append(df)

    if not results:
        return pd.DataFrame(
            columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
        )
    return pd.concat(results, ignore_index=True)


# ---- Yahoo (optional fallback) ----
def _make_session(user_agent: str | None = None) -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": user_agent or settings.YF_USER_AGENT,
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9,de-DE;q=0.8",
            "Connection": "keep-alive",
        "Referer": "https://finance.yahoo.com/",
        }
    )
    retry = Retry(
        total=MAX_RETRIES,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=16)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def _download_single_safe(
    ticker: str, session: requests.Session, start=None, period="1mo"
) -> pd.DataFrame:
    today_utc = pd.Timestamp.utcnow().date()
    for attempt in range(MAX_RETRIES):
        try:
            tk = yf.Ticker(ticker, session=session)
            use_start = start
            use_period = period
            if use_start is not None and pd.to_datetime(use_start).date() >= today_utc:
                use_start = None
                use_period = "7d"
            if use_start is not None:
                hist = tk.history(
                    start=use_start, interval="1d", auto_adjust=False, actions=False, timeout=30
                )
            else:
                hist = tk.history(
                    period=use_period, interval="1d", auto_adjust=False, actions=False, timeout=30
                )
            if hist is not None and not hist.empty:
                df = hist.reset_index().rename(columns={"Date": "date"})
                df["ticker"] = ticker
                colmap = {
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Adj Close": "adj_close",
                    "Volume": "volume",
                }
                df.rename(
                    columns={k: v for k, v in colmap.items() if k in df.columns}, inplace=True
                )
                for c in ("adj_close", "volume"):
                    if c not in df.columns:
                        df[c] = pd.NA
                df["date"] = pd.to_datetime(df["date"]).dt.date
                return df[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]]
            else:
                log.warning(f"{ticker}: no trading data for start date {use_start}")
                return pd.DataFrame(
                    columns=[
                        "date",
                        "ticker",
                        "open",
                        "high",
                        "low",
                        "close",
                        "adj_close",
                        "volume",
                    ]
                )
        except json.JSONDecodeError:
            _retry_sleep(attempt)
            continue
        except HTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status == 429:
                log.warning(f"{ticker}: rate limited (429), backing off")
            _retry_sleep(attempt)
            continue
        except Exception:
            _retry_sleep(attempt)
            continue
    return pd.DataFrame(
        columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    )


def _yahoo_fetch_many(
    tickers: list[str],
    start_map: dict[str, pd.Timestamp] | None = None,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch multiple tickers from Yahoo concurrently."""
    sess = session or _make_session()
    results: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=min(YAHOO_MAX_WORKERS, len(tickers))) as ex:
        futures = {
            ex.submit(
                _download_single_safe,
                t,
                sess,
                start=start_map.get(t) if start_map else None,
                period="1mo",
            ): t
            for t in tickers
        }
        for fut in as_completed(futures):
            df = fut.result()
            if df is not None and not df.empty:
                results.append(df)
    if not results:
        return pd.DataFrame(
            columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
        )
    return pd.concat(results, ignore_index=True)


# ---- Public API ----
def update_prices(db_path: str, tickers: list[str]) -> int:
    """
    Schreibt Daily‑Kurse ins DuckDB‑Schema 'prices'.
    Quelle per ENV (WALLENSTEIN_DATA_SOURCE):
      - 'yahoo'  (default, nur Yahoo)
      - 'stooq'  (Stooq mit Yahoo‑Fallback)
    'hybrid' funktioniert weiterhin als Alias für 'stooq'.
    """
    if not tickers:
        raise ValueError("Keine Ticker übergeben.")

    con = _connect(db_path)
    ensure_tables(con)

    last_map = _latest_dates_per_ticker(con, tickers)
    all_rows: list[pd.DataFrame] = []

    session = None
    if DATA_SOURCE in ("yahoo", "hybrid"):
        session = _make_session()

    # letzter (Börsen-)Tag bis heute (naiv ok, da reine Tagesdaten)
    today = pd.Timestamp.utcnow().normalize()
    last_trading_day = pd.bdate_range(end=today, periods=1)[0].date()

    # Start-Datum je Ticker vorbereiten (None => Voll-Download)
    start_map: dict[str, pd.Timestamp | None] = {}
    for t in tickers:
        ld = last_map.get(t)  # datetime.date oder None
        if pd.notna(ld):
            # ab dem Tag NACH dem letzten gespeicherten Tag laden
            start_map[t] = pd.Timestamp(ld) + pd.Timedelta(days=1)
        else:
            start_map[t] = None

    for i in range(0, len(tickers), CHUNK_SIZE):
        chunk = tickers[i : i + CHUNK_SIZE]

        # Start-Tage, die nach dem letzten Handelstag liegen, überspringen
        chunk_valid: list[str] = []
        for t in chunk:
            s = start_map.get(t)
            if s is not None:
                s_d = pd.to_datetime(s).date()
                if s_d > last_trading_day:
                    continue
            chunk_valid.append(t)
        if not chunk_valid:
            continue

        if DATA_SOURCE == "yahoo":
            df_chunk = _yahoo_fetch_many(chunk_valid, start_map=start_map, session=session)
        else:
            df_chunk = _stooq_fetch_many(chunk_valid, start_map=start_map)

            # Fallback: fehlende Ticker via Yahoo nachladen
            missing = [t for t in chunk_valid if df_chunk[df_chunk["ticker"].eq(t)].empty]
            if missing:
                df_fb = _yahoo_fetch_many(missing, start_map=start_map, session=session)
                if df_chunk is None or df_chunk.empty:
                    df_chunk = df_fb
                elif df_fb is not None and not df_fb.empty:
                    df_chunk = pd.concat([df_chunk, df_fb], ignore_index=True)

        if df_chunk is None or df_chunk.empty:
            continue

        # Nur neue Zeilen (nach last_map[t]) anhängen
        for t in chunk_valid:
            df_t = df_chunk[df_chunk["ticker"].eq(t)]
            if df_t.empty:
                continue
            ld = last_map.get(t)
            if pd.notna(ld):
                ld_d = pd.to_datetime(ld).date()
                df_t = df_t[pd.to_datetime(df_t["date"]).dt.date > ld_d]
            if not df_t.empty:
                all_rows.append(
                    df_t[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]]
                )

    if not all_rows:
        if session is not None:
            session.close()
        con.close()
        return 0

    df_all = pd.concat(all_rows, ignore_index=True)
    df_all["date"] = pd.to_datetime(df_all["date"]).dt.date
    df_all = df_all.sort_values(["ticker", "date"])

    # Append + DISTINCT‑Refresh
    validate_df(df_all, "prices")
    con.execute("INSERT INTO prices SELECT * FROM df_all")
    con.execute(
        """
        CREATE OR REPLACE TABLE prices AS
        SELECT DISTINCT * FROM prices
    """
    )

    n = len(df_all)
    if session is not None:
        session.close()
    con.close()
    return n


def _ensure_fx_table(con: duckdb.DuckDBPyConnection):
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS fx_rates (
            date DATE,
            pair VARCHAR,
            rate_usd_per_eur DOUBLE
        )
    """
    )


def _fx_latest_date(con: duckdb.DuckDBPyConnection, pair: str):
    try:
        row = con.execute("SELECT max(date) FROM fx_rates WHERE pair = ?", [pair]).fetchone()
        return row[0]
    except Exception:
        return None


def _stooq_fetch_fx_eurusd(start: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Holt Daily EURUSD von Stooq (stabil). Stooq liefert 'eurusd' als CSV.
    rate_usd_per_eur := Close (USD je 1 EUR)
    """
    url = "https://stooq.com/q/d/l/?s=eurusd&i=d"
    try:
        r = requests.get(url, timeout=20, headers=STOOQ_HEADERS)
        if not r.ok or not r.text:
            return pd.DataFrame()
        df = pd.read_csv(io.StringIO(r.text))
        if df.empty:
            return pd.DataFrame()
        df = df.rename(columns={"Date": "date", "Close": "close"})
        df["date"] = pd.to_datetime(df["date"]).dt.date
        if start is not None:
            start_d = pd.to_datetime(start).date()
            df = df[df["date"] >= start_d]
        df["pair"] = "EURUSD"
        df["rate_usd_per_eur"] = pd.to_numeric(df["close"], errors="coerce")
        return df[["date", "pair", "rate_usd_per_eur"]].dropna()
    except Exception:
        return pd.DataFrame()


def update_fx_rates(db_path: str) -> int:
    """
    Aktualisiert fx_rates mit EURUSD (Stooq).
    """
    con = _connect(db_path)
    _ensure_fx_table(con)

    last = _fx_latest_date(con, "EURUSD")
    start = None
    if pd.notna(last):
        start = pd.Timestamp(last) + pd.Timedelta(days=1)

    df = _stooq_fetch_fx_eurusd(start=start)
    if df is None or df.empty:
        con.close()
        return 0

    df = df.sort_values("date")
    con.execute("INSERT INTO fx_rates SELECT * FROM df")
    # Dubletten vermeiden
    con.execute(
        """
        CREATE OR REPLACE TABLE fx_rates AS
        SELECT DISTINCT * FROM fx_rates
    """
    )
    n = len(df)
    con.close()
    return n
