"""
Stock price utilities with robust multi-source fallbacks + detailed debug logs.

Primär:
  - Yahoo Chart API (JSON; range=2d&interval=1d)
  - yfinance (raise_errors=False), alternativ 2d/1d
Fallback:
  - Stooq (CSV)
Intraday:
  - Aggregation aus Chart-API (multi-interval/multi-range) für HEUTE

Schreiben:
  - MERGE-UPSERT (wenn verfügbar), sonst DELETE WHERE EXISTS + INSERT
"""

from __future__ import annotations

import io
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

# Optional: curl_cffi (robustere HTTP-Session gg. Blocks/429)
try:
    from curl_cffi import requests as cffi_requests  # type: ignore

    HAVE_CURL_CFFI = True
except Exception:  # pragma: no cover
    cffi_requests = None
    HAVE_CURL_CFFI = False

from wallenstein.config import settings
from wallenstein.db_schema import ensure_tables, validate_df
from wallenstein.db_utils import _ensure_fx_table

# ------------------------------------------------------------------------------
# Konfiguration
# ------------------------------------------------------------------------------

MAX_RETRIES = 6
CHUNK_SIZE = 20
YAHOO_MAX_WORKERS = 3  # moderat halten, reduziert 429-Risiko
REFRESH_SAME_DAY = True  # HEUTE immer aktualisieren (Intraday/Close-Refresh)
# Stooq User-Agent (einige Umgebungen blocken Default-UA)
STOOQ_HEADERS = {"User-Agent": settings.STOOQ_USER_AGENT}
# Datenquelle: 'yahoo' (primär Yahoo) oder 'stooq' (primär Stooq)
DATA_SOURCE = (settings.WALLENSTEIN_DATA_SOURCE or "yahoo").strip().lower()

log = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# DuckDB Helpers
# ------------------------------------------------------------------------------


def _connect(db_path: str) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(db_path)


def _ensure_prices_table(con: duckdb.DuckDBPyConnection):
    # Wenn 'prices' als VIEW existiert: droppen
    row = con.execute(
        """
        SELECT table_type
        FROM information_schema.tables
        WHERE lower(table_name) = 'prices'
        LIMIT 1
        """
    ).fetchone()
    if row and row[0] == "VIEW":
        log.warning("Found VIEW 'prices' -> dropping to create a base table.")
        con.execute("DROP VIEW prices")

    # Tabelle (mit PK) sicherstellen
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
            volume BIGINT,
            PRIMARY KEY(date, ticker)
        )
        """
    )
    try:
        con.execute("CREATE INDEX IF NOT EXISTS prices_ticker_date_idx ON prices(ticker, date)")
    except Exception:
        pass  # ältere DuckDBs ohne Indexunterstützung


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
        m = {row["ticker"]: row["last_date"] for _, row in df.iterrows()}
        return m
    except Exception as e:
        log.debug(f"_latest_dates_per_ticker: query failed: {e}")
        return {}


def _retry_sleep(attempt: int):
    # exponentiell + Jitter
    time.sleep((2**attempt) + random.random())


# ------------------------------------------------------------------------------
# Stooq (CSV)
# ------------------------------------------------------------------------------


def _stooq_symbol(t: str) -> str:
    # Stooq erwartet z.B. nvda.us, amzn.us
    return f"{t.lower()}.us"


def _stooq_fetch_one(
    ticker: str, start: pd.Timestamp | None = None, session: requests.Session | None = None
) -> pd.DataFrame:
    sym = _stooq_symbol(ticker)
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    sess = session or requests
    for attempt in range(MAX_RETRIES):
        try:
            r = sess.get(url, timeout=20, headers=STOOQ_HEADERS)
            if not r.ok or not r.text:
                raise RuntimeError(f"stooq bad response status={r.status_code}")
            df = pd.read_csv(io.StringIO(r.text))
            if df.empty:
                raise RuntimeError("stooq empty dataframe")
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
            df["adj_close"] = pd.NA
            return df[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]]
        except Exception as e:
            log.debug(f"Stooq {_stooq_symbol(ticker)} attempt#{attempt+1} failed: {e}")
            _retry_sleep(attempt)
    return pd.DataFrame(
        columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    )


def _stooq_fetch_many(
    tickers: list[str], start_map: dict[str, pd.Timestamp] | None = None
) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame(
            columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
        )
    results: list[pd.DataFrame] = []

    def _fetch(t: str) -> pd.DataFrame:
        s = start_map.get(t) if start_map else None
        return _stooq_fetch_one(t, start=s)

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


# ------------------------------------------------------------------------------
# HTTP Sessions
# ------------------------------------------------------------------------------


def _make_session(user_agent: str | None = None):
    """
    Erst curl_cffi-Session (impersonate=chrome), sonst std-requests mit Retry.
    """
    if HAVE_CURL_CFFI:
        s = cffi_requests.Session(impersonate="chrome")
        s.headers.update(
            {
                "User-Agent": user_agent or settings.YF_USER_AGENT,
                "Accept": "*/*",
                "Accept-Language": "en-US,en;q=0.9,de-DE;q=0.8",
                "Connection": "keep-alive",
                "Referer": "https://finance.yahoo.com/",
            }
        )
        return s
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


# ------------------------------------------------------------------------------
# Yahoo Chart API (JSON)
# ------------------------------------------------------------------------------


def _chart_api_get_raw(ticker: str, params: dict, session) -> dict | None:
    """Fragt query2/query1 chart v8 ab und liefert einen einzelnen 'result'-Eintrag."""
    for host in ("query2.finance.yahoo.com", "query1.finance.yahoo.com"):
        url = f"https://{host}/v8/finance/chart/{ticker}"
        try:
            r = session.get(
                url, params=params, timeout=20, headers={"Referer": "https://finance.yahoo.com/"}
            )
            if not r.ok:
                log.debug(f"{ticker}: chart API {host} status={r.status_code} params={params}")
                continue
            data = r.json() or {}
            chart = data.get("chart", {})
            if chart.get("error"):
                err = chart["error"]
                code = err.get("code")
                desc = err.get("description")
                log.debug(f"{ticker}: chart.error code={code} desc={desc} params={params}")
            result = chart.get("result")
            if result:
                return result[0]
        except Exception as e:
            log.debug(f"{ticker}: chart API {host} failed: {e} params={params}")
            continue
    return None


def _yahoo_chart_api_daily(ticker: str, session) -> pd.DataFrame:
    """
    Holt gestern+heute als Daily-Bars (range=2d, interval=1d).
    """
    params = {"range": "2d", "interval": "1d", "includePrePost": "true", "events": "div,splits"}
    out = _chart_api_get_raw(ticker, params, session)
    if not out:
        return pd.DataFrame(
            columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
        )
    ts = out.get("timestamp") or []
    q = (out.get("indicators", {}) or {}).get("quote", [{}])[0]
    if not ts or not q:
        return pd.DataFrame(
            columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
        )
    df = pd.DataFrame(
        {
            "dt": pd.to_datetime(ts, unit="s", utc=True),
            "open": q.get("open"),
            "high": q.get("high"),
            "low": q.get("low"),
            "close": q.get("close"),
            "volume": q.get("volume"),
        }
    ).dropna(subset=["open", "high", "low", "close"], how="all")
    if df.empty:
        return pd.DataFrame(
            columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
        )
    try:
        df["dt"] = df["dt"].dt.tz_convert(None)
    except Exception:
        df["dt"] = df["dt"].dt.tz_localize(None)
    df["date"] = df["dt"].dt.date
    df["ticker"] = ticker
    df["adj_close"] = pd.NA
    return df[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]]


def _chart_result_to_intraday_df(ticker: str, out: dict) -> pd.DataFrame:
    if not out:
        return pd.DataFrame(columns=["dt", "open", "high", "low", "close", "volume"])
    ts = out.get("timestamp") or []
    q = (out.get("indicators", {}) or {}).get("quote", [{}])[0]
    if not ts or not q:
        return pd.DataFrame(columns=["dt", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(
        {
            "dt": pd.to_datetime(ts, unit="s", utc=True),
            "open": q.get("open"),
            "high": q.get("high"),
            "low": q.get("low"),
            "close": q.get("close"),
            "volume": q.get("volume"),
        }
    ).dropna(subset=["open", "high", "low", "close"], how="all")
    if df.empty:
        return df
    try:
        df["dt"] = df["dt"].dt.tz_convert(None)
    except Exception:
        df["dt"] = df["dt"].dt.tz_localize(None)
    return df


def _yahoo_chart_api_intraday_to_daily(
    ticker: str,
    session,
    intervals: list[str] | None = None,
    ranges: list[str] | None = None,
) -> pd.DataFrame:
    """
    Aggregiert HEUTE aus Intraday-Daten:
      - probiert mehrere intervals (1m, 2m, 5m, 15m) und ranges (1d, 5d)
      - Open = erstes, High = max, Low = min, Close = letztes, Volume = sum
    """
    if intervals is None:
        intervals = ["1m", "2m", "5m", "15m"]
    if ranges is None:
        ranges = ["1d", "5d"]

    today = pd.Timestamp.utcnow().date()
    for rng in ranges:
        for iv in intervals:
            params = {
                "range": rng,
                "interval": iv,
                "includePrePost": "true",
                "events": "div,splits",
                "corsDomain": "finance.yahoo.com",
            }
            out = _chart_api_get_raw(ticker, params, session)
            df = _chart_result_to_intraday_df(ticker, out)
            if df is None or df.empty:
                log.debug(f"{ticker}: intraday empty range={rng} interval={iv}")
                continue
            dft = df[df["dt"].dt.date.eq(today)]
            if dft.empty:
                log.debug(f"{ticker}: intraday no-today range={rng} interval={iv}")
                continue

            o = float(dft["open"].iloc[0])
            h = float(pd.to_numeric(dft["high"], errors="coerce").max())
            low_val = float(pd.to_numeric(dft["low"], errors="coerce").min())
            c = float(dft["close"].iloc[-1])
            v = int(pd.to_numeric(dft["volume"], errors="coerce").fillna(0).sum())

            outdf = pd.DataFrame(
                [
                    {
                        "date": today,
                        "ticker": ticker,
                        "open": o,
                        "high": h,
                        "low": low_val,
                        "close": c,
                        "adj_close": pd.NA,
                        "volume": v,
                    }
                ]
            )
            log.debug(f"{ticker}: intraday OK range={rng} interval={iv} rows={len(dft)}")
            return outdf

    return pd.DataFrame(
        columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    )


# ------------------------------------------------------------------------------
# Yahoo Download (safe, mehrstufig)
# ------------------------------------------------------------------------------


def _download_single_safe(
    ticker: str,
    session,
    start: pd.Timestamp | None = None,
    period: str = "1mo",
) -> pd.DataFrame:
    """
    Mehrstufiger Yahoo-Download:
      1) Chart-API daily (2d/1d)
      2) yfinance history (raise_errors=False)
      3) yfinance 2d/1d
    """

    def _empty() -> pd.DataFrame:
        return pd.DataFrame(
            columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
        )

    def _norm(df_hist: pd.DataFrame) -> pd.DataFrame:
        if df_hist is None or df_hist.empty:
            return _empty()
        df = df_hist.reset_index().rename(columns={"Date": "date"})
        df["ticker"] = ticker
        colmap = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
        df.rename(columns={k: v for k, v in colmap.items() if k in df.columns}, inplace=True)
        for c in ("adj_close", "volume"):
            if c not in df.columns:
                df[c] = pd.NA
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]]

    today_utc = pd.Timestamp.utcnow().date()
    last_err: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            # (1) Chart-API daily
            df_api = _yahoo_chart_api_daily(ticker, session=session)
            if start is not None and not df_api.empty:
                sd = pd.to_datetime(start).date()
                df_api = df_api[df_api["date"] >= sd]
            if not df_api.empty:
                log.debug(f"{ticker}: chart daily rows={len(df_api)} (attempt={attempt+1})")
                return df_api

            # (2) yfinance history (kein harter Fehler)
            tk = yf.Ticker(ticker, session=session)
            use_start = start
            use_period = period
            if use_start is not None and pd.to_datetime(use_start).date() >= today_utc:
                use_start = None
                use_period = "7d"
            if use_start is not None:
                hist = tk.history(
                    start=use_start,
                    interval="1d",
                    auto_adjust=False,
                    actions=False,
                    timeout=30,
                    raise_errors=False,
                )
            else:
                hist = tk.history(
                    period=use_period,
                    interval="1d",
                    auto_adjust=False,
                    actions=False,
                    timeout=30,
                    raise_errors=False,
                )
            df = _norm(hist)
            if not df.empty:
                log.debug(f"{ticker}: yf.history rows={len(df)} (attempt={attempt+1})")
                return df

            # (3) yfinance 2d/1d
            hist2 = tk.history(
                period="2d",
                interval="1d",
                auto_adjust=False,
                actions=False,
                timeout=30,
                raise_errors=False,
            )
            df2 = _norm(hist2)
            if not df2.empty:
                log.debug(f"{ticker}: yf.history 2d/1d rows={len(df2)} (attempt={attempt+1})")
                return df2

            # leer -> Retry
            raise RuntimeError("empty yahoo response")

        except HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status == 429:
                log.warning(f"{ticker}: skipped due to rate limiting")
                return _empty()
            last_err = e
            log.debug(f"{ticker}: HTTPError on attempt {attempt+1}: {e}")
        except Exception as e:
            last_err = e
            # yfinance-spezifische Fehler: neue Session probieren
            if e.__class__.__name__ in {"YFDataException", "YFRateLimitError"}:
                log.debug(f"{ticker}: {e.__class__.__name__} -> rebuild session")
                try:
                    session.close()
                except Exception:
                    pass
                session = _make_session()
            else:
                log.debug(f"{ticker}: Exception on attempt {attempt+1}: {e}")
        if attempt < MAX_RETRIES - 1:
            _retry_sleep(attempt)
            continue

    if last_err is not None:
        log.warning(f"{ticker}: network error ({last_err.__class__.__name__})")
    return _empty()


def _yahoo_fetch_many(
    tickers: list[str],
    start_map: dict[str, pd.Timestamp] | None = None,
    session=None,
) -> pd.DataFrame:
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


# ------------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------------


def update_prices(db_path: str, tickers: list[str]) -> int:
    """
    Schreibt Daily-Kurse ins DuckDB-Schema 'prices'.
    Quelle per ENV (WALLENSTEIN_DATA_SOURCE):
      - 'yahoo'  (primär Yahoo, Stooq-Fallback)
      - 'stooq'  (primär Stooq, Yahoo-Fallback)
    """
    if not tickers:
        raise ValueError("Keine Ticker übergeben.")

    con = _connect(db_path)
    ensure_tables(con)
    _ensure_prices_table(con)

    log.info("DATA_SOURCE=%s | tickers=%s", DATA_SOURCE, tickers)

    last_map = _latest_dates_per_ticker(con, tickers)
    if last_map:
        log.debug("last_map subset: %s", {k: str(last_map.get(k)) for k in tickers})

    # letzter (Börsen-)Tag bis heute (werktäglich)
    today = pd.Timestamp.utcnow().normalize()
    last_trading_day = pd.bdate_range(end=today, periods=1)[0].date()
    log.debug("last_trading_day=%s (UTC)", last_trading_day)
    if today.date() > last_trading_day:
        con.close()
        log.info("Keine neuen Kursdaten am Wochenende.")
        return 0

    session = _make_session()

    # Start-Datum je Ticker (None => Voll-Download)
    start_map: dict[str, pd.Timestamp | None] = {}
    for t in tickers:
        ld = last_map.get(t)
        if pd.notna(ld):
            start_map[t] = pd.Timestamp(ld) + pd.Timedelta(days=1)
        else:
            start_map[t] = None
    log.debug(
        "start_map: %s",
        {k: (str(v.date()) if v is not None else None) for k, v in start_map.items()},
    )

    all_rows: list[pd.DataFrame] = []

    for i in range(0, len(tickers), CHUNK_SIZE):
        chunk = tickers[i : i + CHUNK_SIZE]
        chunk_valid = [
            t for t in chunk if start_map.get(t) is None or start_map[t].date() <= last_trading_day
        ]
        if not chunk_valid:
            continue
        log.debug("processing chunk=%s", chunk_valid)

        # --- Primärquelle ---
        if DATA_SOURCE == "yahoo":
            df_chunk = _yahoo_fetch_many(chunk_valid, start_map=start_map, session=session)
            log.debug(
                "primary=yahoo rows=%s tickers=%s",
                0 if df_chunk is None or df_chunk.empty else len(df_chunk),
                [] if df_chunk is None or df_chunk.empty else sorted(df_chunk["ticker"].unique()),
            )
            # Fallback Stooq für Ticker ohne Zeilen
            missing = (
                chunk_valid
                if df_chunk is None or df_chunk.empty
                else [t for t in chunk_valid if df_chunk[df_chunk["ticker"].eq(t)].empty]
            )
            if missing:
                log.debug("fallback stooq for: %s", missing)
                df_fb = _stooq_fetch_many(missing, start_map=start_map)
                if df_chunk is None or df_chunk.empty:
                    df_chunk = df_fb
                elif df_fb is not None and not df_fb.empty:
                    df_chunk = pd.concat([df_chunk, df_fb], ignore_index=True)
                log.debug(
                    "after fallback rows=%s tickers=%s",
                    0 if df_chunk is None or df_chunk.empty else len(df_chunk),
                    (
                        []
                        if df_chunk is None or df_chunk.empty
                        else sorted(df_chunk["ticker"].unique())
                    ),
                )
        else:
            df_chunk = _stooq_fetch_many(chunk_valid, start_map=start_map)
            log.debug(
                "primary=stooq rows=%s tickers=%s",
                0 if df_chunk is None or df_chunk.empty else len(df_chunk),
                [] if df_chunk is None or df_chunk.empty else sorted(df_chunk["ticker"].unique()),
            )
            missing = (
                chunk_valid
                if df_chunk is None or df_chunk.empty
                else [t for t in chunk_valid if df_chunk[df_chunk["ticker"].eq(t)].empty]
            )
            if missing:
                log.debug("fallback yahoo for: %s", missing)
                df_fb = _yahoo_fetch_many(missing, start_map=start_map, session=session)
                if df_chunk is None or df_chunk.empty:
                    df_chunk = df_fb
                elif df_fb is not None and not df_fb.empty:
                    df_chunk = pd.concat([df_chunk, df_fb], ignore_index=True)
                log.debug(
                    "after fallback rows=%s tickers=%s",
                    0 if df_chunk is None or df_chunk.empty else len(df_chunk),
                    (
                        []
                        if df_chunk is None or df_chunk.empty
                        else sorted(df_chunk["ticker"].unique())
                    ),
                )

        # Wenn immer noch nichts da: Intraday->Daily für den ganzen Chunk
        if df_chunk is None or df_chunk.empty:
            log.debug("primary+fallback empty -> intraday aggregation for all in chunk")
            intraday_rows = [
                _yahoo_chart_api_intraday_to_daily(t, session=session) for t in chunk_valid
            ]
            intraday_rows = [d for d in intraday_rows if d is not None and not d.empty]
            if intraday_rows:
                df_chunk = pd.concat(intraday_rows, ignore_index=True)
            else:
                log.debug("intraday aggregation empty for chunk=%s", chunk_valid)
                continue

        # Nur neue Zeilen (bzw. Same-Day Refresh) je Ticker
        updated_tickers = []
        for t in chunk_valid:
            df_t = df_chunk[df_chunk["ticker"].eq(t)]
            if df_t.empty:
                # ultima ratio: pro Ticker Intraday
                df_t = _yahoo_chart_api_intraday_to_daily(t, session=session)
                if df_t is None or df_t.empty:
                    continue
            ld = last_map.get(t)
            if pd.notna(ld):
                ld_d = pd.to_datetime(ld).date()
                mask_new = pd.to_datetime(df_t["date"]).dt.date > ld_d
                if REFRESH_SAME_DAY and ld_d == last_trading_day:
                    mask_new = mask_new | (pd.to_datetime(df_t["date"]).dt.date == last_trading_day)
                df_t = df_t[mask_new]
            if not df_t.empty:
                all_rows.append(
                    df_t[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]]
                )
                updated_tickers.append(t)

        log.debug("tickers with new/refresh rows in chunk: %s", updated_tickers)

    if not all_rows:
        try:
            session.close()
        except Exception:
            pass
        con.close()
        log.info("Keine neuen Kurszeilen.")
        return 0

    df_all = pd.concat(all_rows, ignore_index=True)
    df_all["date"] = pd.to_datetime(df_all["date"]).dt.date
    df_all = df_all.sort_values(["ticker", "date"]).dropna(subset=["date", "ticker"])

    # Validierung
    validate_df(df_all, "prices")
    log.debug("total new/refresh rows=%s", len(df_all))

    # In temp table
    con.execute("CREATE TEMP TABLE tmp_prices AS SELECT * FROM df_all")

    # Upsert: MERGE (wenn möglich) -> sonst DELETE+INSERT
    try:
        con.execute(
            """
            MERGE INTO prices AS p
            USING tmp_prices AS d
            ON p.date = d.date AND p.ticker = d.ticker
            WHEN MATCHED THEN UPDATE SET
                open = d.open, high = d.high, low = d.low, close = d.close,
                adj_close = d.adj_close, volume = d.volume
            WHEN NOT MATCHED THEN INSERT (date, ticker, open, high, low, close, adj_close, volume)
            VALUES (d.date, d.ticker, d.open, d.high, d.low, d.close, d.adj_close, d.volume)
        """
        )
        log.debug("UPSERT via MERGE successful.")
    except Exception as e:
        log.warning(
            f"MERGE not supported/failed ({e.__class__.__name__}): fallback to DELETE+INSERT."
        )
        con.execute("BEGIN")
        con.execute(
            """
            DELETE FROM prices
            WHERE EXISTS (
                SELECT 1 FROM tmp_prices d
                WHERE d.date = prices.date
                  AND d.ticker = prices.ticker
            )
        """
        )
        con.execute("INSERT INTO prices SELECT * FROM tmp_prices")
        con.execute("COMMIT")
        log.debug("UPSERT via DELETE+INSERT done.")

    n = len(df_all)
    try:
        session.close()
    except Exception:
        pass
    con.close()
    log.info("Kursdaten aktualisiert/aktualisiert: +%s Zeilen", n)
    return n


# ------------------------------------------------------------------------------
# FX: EURUSD via Stooq
# ------------------------------------------------------------------------------

def _fx_latest_date(con: duckdb.DuckDBPyConnection, pair: str):
    try:
        row = con.execute("SELECT max(date) FROM fx_rates WHERE pair = ?", [pair]).fetchone()
        return row[0]
    except Exception:
        return None


def _stooq_fetch_fx_eurusd(start: pd.Timestamp | None = None) -> pd.DataFrame:
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
    except Exception as e:
        log.debug(f"stooq EURUSD failed: {e}")
        return pd.DataFrame()


def update_fx_rates(db_path: str) -> int:
    con = _connect(db_path)
    _ensure_fx_table(con)

    last = _fx_latest_date(con, "EURUSD")
    start = pd.Timestamp(last) + pd.Timedelta(days=1) if pd.notna(last) else None

    df = _stooq_fetch_fx_eurusd(start=start)
    if df is None or df.empty:
        con.close()
        log.info("FX-Update: keine neuen EURUSD-Zeilen")
        return 0

    df = df.sort_values("date").drop_duplicates(subset=["date", "pair"])
    con.execute("CREATE TEMP TABLE tmp_fx AS SELECT * FROM df")

    # Upsert mit MERGE, Fallback DELETE+INSERT
    try:
        con.execute(
            """
            MERGE INTO fx_rates AS f
            USING tmp_fx AS d
            ON f.date = d.date AND f.pair = d.pair
            WHEN MATCHED THEN UPDATE SET rate_usd_per_eur = d.rate_usd_per_eur
            WHEN NOT MATCHED THEN INSERT (date, pair, rate_usd_per_eur)
            VALUES (d.date, d.pair, d.rate_usd_per_eur)
        """
        )
        log.debug("FX UPSERT via MERGE successful.")
    except Exception as e:
        log.warning(f"FX MERGE not supported/failed ({e.__class__.__name__}) -> DELETE+INSERT.")
        con.execute("BEGIN")
        con.execute(
            """
            DELETE FROM fx_rates
            WHERE EXISTS (
                SELECT 1 FROM tmp_fx d
                WHERE d.date = fx_rates.date
                  AND d.pair = fx_rates.pair
            )
        """
        )
        con.execute("INSERT INTO fx_rates SELECT * FROM tmp_fx")
        con.execute("COMMIT")
        log.debug("FX UPSERT via DELETE+INSERT done.")

    n = len(df)
    con.close()
    log.info("FX-Update: +%s neue EURUSD-Zeilen", n)
    return n
