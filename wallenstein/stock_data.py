import os
import io
import json
import time
import logging
from typing import List, Optional
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import duckdb
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yfinance as yf

# ---- Configuration ----
MAX_RETRIES = 3
CHUNK_SIZE = 20

# Datenquelle: stooq (default, mit Yahoo-Fallback) | yahoo
# 'hybrid' bleibt aus Kompatibilitätsgründen als Alias zu stooq bestehen
DATA_SOURCE = os.getenv("WALLENSTEIN_DATA_SOURCE", "stooq").strip().lower()

log = logging.getLogger(__name__)

# ---- DuckDB Helpers ----
def _connect(db_path: str) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(db_path)

def _ensure_prices_table(con: duckdb.DuckDBPyConnection):
    # Typ des Objekts "prices" ermitteln: None | 'BASE TABLE' | 'VIEW'
    row = con.execute("""
        SELECT table_type
        FROM information_schema.tables
        WHERE lower(table_name) = 'prices'
        LIMIT 1
    """).fetchone()
    obj_type = row[0] if row else None

    # Nur wenn es wirklich eine VIEW ist, droppen
    if obj_type == 'VIEW':
        con.execute("DROP VIEW prices")

    # Tabelle sicherstellen (no-op, falls schon vorhanden)
    con.execute("""
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
    """)

def _latest_dates_per_ticker(con: duckdb.DuckDBPyConnection, tickers: List[str]):
    if not tickers:
        return {}
    q = """
        SELECT ticker, max(date) AS last_date
        FROM prices
        WHERE ticker IN ({})
        GROUP BY ticker
    """.format(", ".join("'" + t + "'" for t in tickers))
    try:
        df = con.execute(q).fetchdf()
        return {row["ticker"]: row["last_date"] for _, row in df.iterrows()}
    except Exception:
        return {}

def _retry_sleep(attempt: int):
    time.sleep(1.0 * (2 ** attempt))  # 1s, 2s, 4s

# ---- Stooq (stable, CSV) ----
def _stooq_symbol(t: str) -> str:
    # Stooq erwartet z.B. nvda.us, amzn.us, smci.us
    return f"{t.lower()}.us"

def _stooq_fetch_one(ticker: str,
                     start: Optional[pd.Timestamp] = None,
                     session: Optional[requests.Session] = None) -> pd.DataFrame:
    """Fetch a single ticker from Stooq.

    A session can be supplied to reuse connections when called repeatedly.
    """
    sym = _stooq_symbol(ticker)
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    sess = session or requests
    try:
        r = sess.get(url, timeout=20)
        if not r.ok or not r.text:
            return pd.DataFrame()
        df = pd.read_csv(io.StringIO(r.text))
        if df.empty:
            return pd.DataFrame()
        # Stooq Spalten: Date,Open,High,Low,Close,Volume
        df = df.rename(columns={
            "Date": "date", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume"
        })
        df["ticker"] = ticker
        df["date"] = pd.to_datetime(df["date"]).dt.date
        if start is not None:
            df = df[pd.to_datetime(df["date"]) >= pd.to_datetime(start)]
        # adj_close nicht vorhanden
        df["adj_close"] = pd.NA
        return df[["date","ticker","open","high","low","close","adj_close","volume"]]
    except Exception:
        return pd.DataFrame()


def _stooq_fetch_many(tickers: List[str],
                      start: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Fetch multiple tickers from Stooq concurrently."""
    if not tickers:
        return pd.DataFrame(columns=["date","ticker","open","high","low","close","adj_close","volume"])

    results: List[pd.DataFrame] = []

    def _fetch(t: str) -> pd.DataFrame:
        return _stooq_fetch_one(t, start=start)

    with ThreadPoolExecutor(max_workers=min(5, len(tickers))) as ex:
        futures = {ex.submit(_fetch, t): t for t in tickers}
        for fut in as_completed(futures):
            df = fut.result()
            if not df.empty:
                results.append(df)

    if not results:
        return pd.DataFrame(columns=["date","ticker","open","high","low","close","adj_close","volume"])
    return pd.concat(results, ignore_index=True)

# ---- Yahoo (optional fallback) ----
def _make_session(user_agent: Optional[str] = None) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": user_agent or os.getenv(
            "YF_USER_AGENT",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9,de-DE;q=0.8",
        "Connection": "keep-alive",
        "Referer": "https://finance.yahoo.com/",
    })
    retry = Retry(
        total=3,
        backoff_factor=1.2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=16)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

def _download_single_safe(ticker: str, session: requests.Session, start=None, period="1mo") -> pd.DataFrame:
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
                hist = tk.history(start=use_start, interval="1d", auto_adjust=False, actions=False, timeout=30)
            else:
                hist = tk.history(period=use_period, interval="1d", auto_adjust=False, actions=False, timeout=30)
            if hist is not None and not hist.empty:
                df = hist.reset_index().rename(columns={"Date": "date"})
                df["ticker"] = ticker
                colmap = {"Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume"}
                df.rename(columns={k:v for k,v in colmap.items() if k in df.columns}, inplace=True)
                for c in ("adj_close","volume"):
                    if c not in df.columns:
                        df[c] = pd.NA
                df["date"] = pd.to_datetime(df["date"]).dt.date
                return df[["date","ticker","open","high","low","close","adj_close","volume"]]
            else:
                log.warning(f"{ticker}: no trading data for start date {use_start}")
                return pd.DataFrame(columns=["date","ticker","open","high","low","close","adj_close","volume"])
        except json.JSONDecodeError:
            _retry_sleep(attempt); continue
        except Exception:
            _retry_sleep(attempt); continue
    return pd.DataFrame(columns=["date","ticker","open","high","low","close","adj_close","volume"])

def _yahoo_fetch_many(tickers: List[str], start: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    session = _make_session()
    results = []
    with ThreadPoolExecutor(max_workers=min(5, len(tickers))) as ex:
        futures = {ex.submit(_download_single_safe, t, session, start=start, period="1mo"): t for t in tickers}
        for fut in as_completed(futures):
            df = fut.result()
            if df is not None and not df.empty:
                results.append(df)
    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)

# ---- Public API ----
def update_prices(db_path: str, tickers: List[str]) -> int:
    """
    Schreibt Daily‑Kurse ins DuckDB‑Schema 'prices'.
    Quelle per ENV (WALLENSTEIN_DATA_SOURCE):
      - 'stooq'  (default, Stooq mit Yahoo‑Fallback)
      - 'yahoo'  (nur Yahoo; nicht empfohlen bei deinen Logs)
    'hybrid' funktioniert weiterhin als Alias für 'stooq'.
    Analysten-Daten sind komplett deaktiviert.
    """
    if not tickers:
        raise ValueError("Keine Ticker übergeben.")

    con = _connect(db_path)
    _ensure_prices_table(con)

    last_map = _latest_dates_per_ticker(con, tickers)
    all_rows = []

    session = None
    if DATA_SOURCE in ("yahoo", "hybrid"):
        session = _make_session()

    today = pd.Timestamp.utcnow().normalize()
    last_trading_day = pd.bdate_range(end=today, periods=1)[0]

    for i in range(0, len(tickers), CHUNK_SIZE):
        chunk = tickers[i:i + CHUNK_SIZE]
        for t in chunk:
            ld = last_map.get(t)
            start = pd.to_datetime(ld).date() + timedelta(days=1) if pd.notna(ld) else None
            if start is not None and pd.to_datetime(start) > last_trading_day:
                continue
            df_t = pd.DataFrame()
            if DATA_SOURCE in ("stooq", "hybrid", ""):
                df_t = _stooq_fetch_one(t, start=start)

            if DATA_SOURCE == "hybrid" and (df_t is None or df_t.empty):
                df_t = _download_single_safe(t, session, start=start, period="1mo")

            if DATA_SOURCE == "yahoo":
                df_t = _download_single_safe(t, session, start=start, period="1mo")

        if DATA_SOURCE == "yahoo":
            df_chunk = _yahoo_fetch_many(chunk, start=start)
        else:
            df_chunk = _stooq_fetch_many(chunk, start=start)

            # Fallback: fehlende Ticker via Yahoo nachladen
            missing = [t for t in chunk if df_chunk[df_chunk["ticker"].eq(t)].empty]
            if missing:
                df_fb = _yahoo_fetch_many(missing, start=start)
                if df_chunk is None or df_chunk.empty:
                    df_chunk = df_fb
                elif df_fb is not None and not df_fb.empty:
                    df_chunk = pd.concat([df_chunk, df_fb], ignore_index=True)

            if df_t is None or df_t.empty:
                continue

            if pd.notna(ld):
                df_t = df_t[pd.to_datetime(df_t["date"]) > pd.to_datetime(ld)]
            if not df_t.empty:
                all_rows.append(df_t[["date","ticker","open","high","low","close","adj_close","volume"]])

    if not all_rows:
        if session is not None:
            session.close()
        con.close()
        return 0

    df_all = pd.concat(all_rows, ignore_index=True)
    df_all["date"] = pd.to_datetime(df_all["date"]).dt.date
    df_all = df_all.sort_values(["ticker", "date"])

    # Append + DISTINCT‑Refresh
    con.execute("INSERT INTO prices SELECT * FROM df_all")
    con.execute("""
        CREATE OR REPLACE TABLE prices AS
        SELECT DISTINCT * FROM prices
    """)

    n = len(df_all)
    if session is not None:
        session.close()
    con.close()
    return n
def _ensure_fx_table(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS fx_rates (
            date DATE,
            pair VARCHAR,
            rate_usd_per_eur DOUBLE
        )
    """)

def _fx_latest_date(con, pair: str):
    try:
        row = con.execute("SELECT max(date) FROM fx_rates WHERE pair = ?", [pair]).fetchone()
        return row[0]
    except Exception:
        return None

def _stooq_fetch_fx_eurusd(start: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Holt Daily EURUSD von Stooq (stabil). Stooq liefert 'eurusd' als CSV.
    rate_usd_per_eur := Close (USD je 1 EUR)
    """
    url = "https://stooq.com/q/d/l/?s=eurusd&i=d"
    try:
        r = requests.get(url, timeout=20)
        if not r.ok or not r.text:
            return pd.DataFrame()
        df = pd.read_csv(io.StringIO(r.text))
        if df.empty:
            return pd.DataFrame()
        df = df.rename(columns={"Date": "date", "Close": "close"})
        df["date"] = pd.to_datetime(df["date"]).dt.date
        if start is not None:
            df = df[pd.to_datetime(df["date"]) >= pd.to_datetime(start)]
        df["pair"] = "EURUSD"
        df["rate_usd_per_eur"] = pd.to_numeric(df["close"], errors="coerce")
        return df[["date", "pair", "rate_usd_per_eur"]].dropna()
    except Exception:
        return pd.DataFrame()

def update_fx_rates(db_path: str) -> int:
    """
    Aktualisiert fx_rates mit EURUSD (Stooq).
    """
    con = _connect(db_path)  # _connect hast du ja schon in stock_data.py
    _ensure_fx_table(con)

    last = _fx_latest_date(con, "EURUSD")
    start = None
    if pd.notna(last):
        start = pd.to_datetime(last).date() + pd.Timedelta(days=1)

    df = _stooq_fetch_fx_eurusd(start=start)
    if df is None or df.empty:
        con.close()
        return 0

    df = df.sort_values("date")
    con.execute("INSERT INTO fx_rates SELECT * FROM df")
    # Dubletten vermeiden
    con.execute("""
        CREATE OR REPLACE TABLE fx_rates AS
        SELECT DISTINCT * FROM fx_rates
    """)
    n = len(df)
    con.close()
    return n
