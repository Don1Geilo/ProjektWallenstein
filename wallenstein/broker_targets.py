# wallenstein/broker_targets.py
from __future__ import annotations
import os
import time
import re
import logging
from typing import Any, Dict, List, Optional

import requests
import yfinance as yf

log = logging.getLogger("wallenstein.targets")

# -------- Konfiguration / Pfad für DB-Fallback --------
DB_PATH = os.getenv("WALLENSTEIN_DB_PATH", "data/wallenstein.duckdb")

# -------- Retry-Session für yfinance & HTTP --------
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def _build_retry_session(total=5, backoff=0.6) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=total,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    return s

_YF_SESSION = _build_retry_session()

# ---------------- Helpers ----------------
def _sf(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None and str(v).strip() != "" else None
    except Exception:
        return None

def _pos(x: Optional[float]) -> Optional[float]:
    return x if (x is not None and x > 0) else None

def _pick_col(cols: List[str], must: str, alt: List[str]=[]) -> Optional[str]:
    """Spalte finden, die 'price' & 'target' + (must|alt) enthält (case-insensitive)."""
    m = must.lower()
    for c in cols:
        cl = c.lower()
        if "price" in cl and "target" in cl and (m in cl or any(a.lower() in cl for a in alt)):
            return c
    return None

def _merge(pref: Dict[str, Optional[float|str]], alt: Dict[str, Optional[float|str]]) -> Dict[str, Optional[float|str]]:
    """Aus pref nehmen, Lücken aus alt füllen."""
    m = dict(pref)
    for k, v in alt.items():
        if m.get(k) in (None, "", 0, 0.0):
            m[k] = v
    return m

# ---------------- DB-Fallback (letzte Werte) ----------------
def _load_last_targets(db_path: str, ticker: str) -> Dict[str, Optional[float|str]]:
    """
    Holt letzten Snapshot aus DuckDB, wenn vorhanden.
    Kein harter Import auf unser DB-Modul, um Zyklen zu vermeiden.
    """
    try:
        import duckdb  # lazy import
        if not os.path.exists(db_path):
            return {}
        q = """
        SELECT target_mean, target_high, target_low, rec_mean, rec_text
        FROM broker_targets
        WHERE ticker = ?
        ORDER BY fetched_at_utc DESC
        LIMIT 1
        """
        with duckdb.connect(db_path, read_only=True) as con:
            row = con.execute(q, [ticker]).fetchone()
        if not row:
            return {}
        return {
            "target_mean": row[0],
            "target_high": row[1],
            "target_low":  row[2],
            "rec_mean":    row[3],
            "rec_text":    row[4],
        }
    except Exception as e:
        log.debug(f"[{ticker}] DB-Fallback not available: {e}")
        return {}

# ---------------- Source A: Yahoo v6 (quote) ----------------
def _yahoo_v6_quote(ticker: str, timeout: float=8.0) -> Dict[str, Optional[float|str]]:
    url = "https://query2.finance.yahoo.com/v6/finance/quote"
    params = {"symbols": ticker}
    headers = {"User-Agent": "Mozilla/5.0"}
    out = {"mean": None, "high": None, "low": None, "rec_mean": None, "rec_text": None}
    try:
        r = _YF_SESSION.get(url, params=params, headers=headers, timeout=timeout)
        log.info(f"[{ticker}] v6 status={r.status_code}")
        if r.status_code != 200:
            return out
        res = (r.json() or {}).get("quoteResponse", {}).get("result", [])
        if not res:
            return out
        q = res[0]
        out["mean"] = _pos(_sf(q.get("targetMeanPrice")))
        out["high"] = _pos(_sf(q.get("targetHighPrice")))
        out["low"]  = _pos(_sf(q.get("targetLowPrice")))
        rm = _sf(q.get("recommendationMean")); rk = q.get("recommendationKey")
        out["rec_mean"] = rm
        out["rec_text"] = (
            "Strong Buy" if rm is not None and rm < 1.5 else
            "Buy"        if rm is not None and rm < 2.5 else
            "Hold"       if rm is not None and rm < 3.5 else
            "Sell"       if rm is not None and rm < 4.5 else
            (rk or "").replace("_"," ").title() or None
        )
        log.info(f"[{ticker}] v6 targets -> mean={out['mean']} high={out['high']} low={out['low']} rec_text={out['rec_text']}")
    except Exception as e:
        log.warning(f"[{ticker}] v6 exception: {e}")
    return out

# ---------------- Source B: yfinance.analysis (alt/neu kompatibel) ----------------
def _analysis_targets(tk: yf.Ticker, ticker: str) -> Dict[str, Optional[float]]:
    out = {"mean": None, "high": None, "low": None}
    try:
        # Kompatibilität: manche Versionen haben .analysis (Property), neuere .get_analysis()
        df = None
        if hasattr(tk, "get_analysis") and callable(getattr(tk, "get_analysis")):
            df = tk.get_analysis()
        elif hasattr(tk, "analysis"):
            df = tk.analysis
        if df is None or getattr(df, "empty", True):
            log.info(f"[{ticker}] analysis: empty")
            return out
        cols = [str(c) for c in df.columns]
        log.info(f"[{ticker}] analysis columns: {cols}")
        c_mean = _pick_col(cols, "mean", ["avg","average"])
        c_high = _pick_col(cols, "high", [])
        c_low  = _pick_col(cols, "low", [])
        row = df.iloc[-1]
        if c_mean: out["mean"] = _pos(_sf(row[c_mean]))
        if c_high: out["high"] = _pos(_sf(row[c_high]))
        if c_low:  out["low"]  = _pos(_sf(row[c_low]))
        log.info(f"[{ticker}] analysis targets -> mean={out['mean']} high={out['high']} low={out['low']}")
    except Exception as e:
        log.warning(f"[{ticker}] analysis exception: {e}")
    return out

# ---------------- Source C: Yahoo v10 (financialData) ----------------
def _yahoo_v10_financialdata(ticker: str, timeout: float=8.0) -> Dict[str, Optional[float|str]]:
    url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
    params = {"modules": "financialData"}
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    out = {"mean": None, "high": None, "low": None, "rec_mean": None, "rec_text": None}
    try:
        r = _YF_SESSION.get(url, params=params, headers=headers, timeout=timeout)
        log.info(f"[{ticker}] v10 status={r.status_code}")
        if r.status_code != 200:
            return out
        res = (r.json() or {}).get("quoteSummary", {}).get("result", [])
        if not res:
            return out
        fin = res[0].get("financialData", {}) or {}
        out["mean"] = _pos(_sf((fin.get("targetMeanPrice") or {}).get("raw")))
        out["high"] = _pos(_sf((fin.get("targetHighPrice") or {}).get("raw")))
        out["low"]  = _pos(_sf((fin.get("targetLowPrice")  or {}).get("raw")))
        rm = _sf((fin.get("recommendationMean") or {}).get("raw")); rk = fin.get("recommendationKey")
        out["rec_mean"] = rm
        out["rec_text"] = (
            "Strong Buy" if rm is not None and rm < 1.5 else
            "Buy"        if rm is not None and rm < 2.5 else
            "Hold"       if rm is not None and rm < 3.5 else
            "Sell"       if rm is not None and rm < 4.5 else
            (rk or "").replace("_"," ").title() or None
        )
        log.info(f"[{ticker}] v10 targets -> mean={out['mean']} high={out['high']} low={out['low']} rec_text={out['rec_text']}")
    except Exception as e:
        log.warning(f"[{ticker}] v10 exception: {e}")
    return out

# ---------------- Recommendations (Text + Counts) ----------------
def _latest_reco_text(tk: yf.Ticker, ticker: str) -> Optional[str]:
    try:
        df = None
        if hasattr(tk, "get_recommendations") and callable(getattr(tk, "get_recommendations")):
            df = tk.get_recommendations()
        elif hasattr(tk, "recommendations"):
            df = tk.recommendations
        if df is None or getattr(df, "empty", True):
            log.info(f"[{ticker}] recommendations: empty"); return None
        cand = next((c for c in df.columns if re.search(r"\bto[_ ]?grade\b", str(c), re.I)), None)
        if cand:
            last = str(df.iloc[-1][cand]).strip()
            log.info(f"[{ticker}] recommendations last '{cand}' = {last}")
            return last or None
        grade_col = next((c for c in df.columns if re.search(r"grade", str(c), re.I)), None)
        if grade_col:
            vc = df.tail(30)[grade_col].astype(str).str.strip().value_counts()
            return str(vc.index[0]) if not vc.empty else None
    except Exception as e:
        log.warning(f"[{ticker}] recommendations exception: {e}")
    return None

def _norm_grade(s: str) -> Optional[str]:
    s = s.strip().lower().replace("_"," ").replace("-"," ")
    s = re.sub(r"\s+"," ",s)
    if "strong" in s and "buy" in s: return "strong_buy"
    if s=="buy" or ("buy" in s and "strong" not in s): return "buy"
    if "hold" in s: return "hold"
    if "strong" in s and "sell" in s: return "strong_sell"
    if s=="sell" or ("sell" in s and "strong" not in s): return "sell"
    return None

def _grade_counts(tk: yf.Ticker, ticker: str, window: int=90) -> Dict[str, Optional[int]]:
    out = {"strong_buy": None, "buy": None, "hold": None, "sell": None, "strong_sell": None}
    try:
        df = None
        if hasattr(tk, "get_recommendations") and callable(getattr(tk, "get_recommendations")):
            df = tk.get_recommendations()
        elif hasattr(tk, "recommendations"):
            df = tk.recommendations
        if df is None or getattr(df, "empty", True): return out
        tail = df.tail(window).copy()
        grade_col = next((c for c in tail.columns if re.search(r"grade", str(c), re.I)), None)
        if not grade_col: return out
        grades = tail[grade_col].astype(str).dropna().map(_norm_grade).dropna()
        if grades.empty: return out
        vc = grades.value_counts()
        for k in out.keys(): out[k] = int(vc.get(k, 0))
    except Exception as e:
        log.warning(f"[{ticker}] grade_counts exception: {e}")
    return out

# ---------------- Public API ----------------
def fetch_broker_snapshot(ticker: str, *, sleep_after: float=0.4) -> Dict[str, Any]:
    """
    Robust:
      1) Yahoo v6 JSON
      2) yfinance analysis (alt/neu)
      3) Yahoo v10 JSON
      4) DB-Fallback (letzte Werte), falls alles leer
    + rec_text & Counts aus recommendations
    """
    now = int(time.time())
    tk = yf.Ticker(ticker, session=_YF_SESSION)

    # A) v6 zuerst (meist stabil)
    v6 = _yahoo_v6_quote(ticker)

    # B) analysis (füllt Lücken)
    ana = _analysis_targets(tk, ticker)
    merged = _merge(
        {"mean": v6["mean"], "high": v6["high"], "low": v6["low"], "rec_mean": v6["rec_mean"], "rec_text": v6["rec_text"]},
        {"mean": ana["mean"], "high": ana["high"], "low": ana["low"]}
    )

    # C) v10 nur, wenn noch Lücken
    if merged["mean"] is None or merged["high"] is None or merged["low"] is None or merged["rec_text"] is None:
        v10 = _yahoo_v10_financialdata(ticker)
        merged = _merge(merged, v10)

    # D) Empfehlung & Counts bevorzugt aus yfinance.recommendations (menschlicher)
    reco_txt = _latest_reco_text(tk, ticker) or merged.get("rec_text")
    counts   = _grade_counts(tk, ticker, window=90)

    # E) Sanity: nur positive Werte
    for f in ("mean","high","low"):
        merged[f] = _pos(_sf(merged.get(f)))

    # F) DB-Fallback (falls alles leer)
    if merged["mean"] is None and merged["high"] is None and merged["low"] is None:
        prev = _load_last_targets(DB_PATH, ticker)
        if prev:
            merged["mean"] = merged["mean"] or prev.get("target_mean")
            merged["high"] = merged["high"] or prev.get("target_high")
            merged["low"]  = merged["low"]  or prev.get("target_low")
            merged["rec_mean"] = merged.get("rec_mean") or prev.get("rec_mean")
            reco_txt = reco_txt or prev.get("rec_text")

    if sleep_after:
        time.sleep(sleep_after)

    return {
        "ticker": ticker,
        "target_mean": merged.get("mean"),
        "target_high": merged.get("high"),
        "target_low":  merged.get("low"),
        "rec_mean":    _sf(merged.get("rec_mean")),
        "rec_text":    reco_txt,
        "strong_buy": counts["strong_buy"], "buy": counts["buy"],
        "hold": counts["hold"], "sell": counts["sell"], "strong_sell": counts["strong_sell"],
        "source": "yahoo.v6 + yf.analysis + yahoo.v10 + yf.recs (+db fallback)",
        "fetched_at_utc": now,
    }

def fetch_many(tickers: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for t in tickers:
        try:
            out.append(fetch_broker_snapshot(t, sleep_after=0.5))
        except Exception as e:
            out.append({
                "ticker": t,
                "error": str(e),
                "source": "agent-broker-targets",
                "fetched_at_utc": int(time.time()),
            })
            time.sleep(0.6)
    return out
