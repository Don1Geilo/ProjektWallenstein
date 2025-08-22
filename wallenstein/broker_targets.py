# wallenstein/broker_targets.py
from __future__ import annotations

"""Fetch analyst price targets and recommendations using the Finnhub API."""

import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger("wallenstein.targets")

FINNHUB_TOKEN = os.getenv("FINNHUB_TOKEN", "demo")
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"


class FinnhubResponseError(Exception):
    """Raised when Finnhub returns a non-JSON response."""

    def __init__(self, status: int, snippet: str):
        super().__init__(f"HTTP {status}: {snippet}")
        self.status = status
        self.snippet = snippet


# ---------------------------------------------------------------------------
# HTTP session with retry
# ---------------------------------------------------------------------------
def _build_session(total: int = 3, backoff: float = 0.5) -> requests.Session:
    """Return a requests session configured with a retry strategy."""

    session = requests.Session()
    retry = Retry(
        total=total,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    return session


_SESSION = _build_session()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _sf(v: Any) -> Optional[float]:
    """Convert value to float if possible."""

    try:
        return float(v) if v is not None and str(v).strip() != "" else None
    except Exception:
        return None


def _pos(x: Optional[float]) -> Optional[float]:
    """Return value only if positive."""

    return x if (x is not None and x > 0) else None


def _finnhub_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Perform a GET request against the Finnhub API and return JSON."""

    params = dict(params)
    params["token"] = FINNHUB_TOKEN
    url = f"{FINNHUB_BASE_URL}/{path}"
    r = _SESSION.get(url, params=params, timeout=10.0)
    r.raise_for_status()
    try:
        return r.json()
    except ValueError as e:
 codex/normalize-and-log-finnhub-response-snippet-ggbtlc
        snippet = " ".join(r.text.split())[:200]  # normalize and truncate

        snippet = " ".join(r.text.split())[:200]
 main
        log.error("Finnhub JSON decode failed [%s]: %s", r.status_code, snippet)
        raise FinnhubResponseError(r.status_code, snippet) from e


# ---------------------------------------------------------------------------
# Finnhub specific helpers
# ---------------------------------------------------------------------------
def _price_targets(ticker: str) -> Dict[str, Optional[float]]:
    """Return mean/high/low analyst price targets for ``ticker``."""

    out = {"mean": None, "high": None, "low": None}
    try:
        data = _finnhub_get("price-target", {"symbol": ticker})
        out["mean"] = _pos(_sf(data.get("targetMean")))
        out["high"] = _pos(_sf(data.get("targetHigh")))
        out["low"] = _pos(_sf(data.get("targetLow")))
    except FinnhubResponseError as e:
        log.warning(f"[{ticker}] price-target error: {e}")
    except Exception as e:
        log.warning(f"[{ticker}] price-target error: {e}")
    return out


def _recommendation_counts(ticker: str) -> Dict[str, Optional[int]]:
    """Return latest analyst recommendation counts for ``ticker``."""

    out = {"strong_buy": None, "buy": None, "hold": None, "sell": None, "strong_sell": None}
    try:
        data = _finnhub_get("stock/recommendation", {"symbol": ticker})
        latest = data[0] if isinstance(data, list) and data else {}
        out["strong_buy"] = latest.get("strongBuy")
        out["buy"] = latest.get("buy")
        out["hold"] = latest.get("hold")
        out["sell"] = latest.get("sell")
        out["strong_sell"] = latest.get("strongSell")
    except FinnhubResponseError as e:
        log.warning(f"[{ticker}] recommendation error: {e}")
    except Exception as e:
        log.warning(f"[{ticker}] recommendation error: {e}")
    return out


def _compute_rec_mean(counts: Dict[str, Optional[int]]) -> Optional[float]:
    """Calculate a mean recommendation score from analyst counts."""

    weights = {"strong_buy": 1, "buy": 2, "hold": 3, "sell": 4, "strong_sell": 5}
    total = sum(v for v in counts.values() if isinstance(v, (int, float)))
    if not total:
        return None
    score = sum(weights[k] * (counts.get(k) or 0) for k in weights)
    return score / total


def _compute_rec_text(mean: Optional[float]) -> Optional[str]:
    """Translate recommendation mean to a human readable text."""

    if mean is None:
        return None
    if mean < 1.5:
        return "Strong Buy"
    if mean < 2.5:
        return "Buy"
    if mean < 3.5:
        return "Hold"
    if mean < 4.5:
        return "Sell"
    return "Strong Sell"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fetch_broker_snapshot(ticker: str, *, sleep_after: float = 0.4) -> Dict[str, Any]:
    """Fetch a broker snapshot for a single ``ticker`` using Finnhub."""

    now = int(time.time())
    targets = _price_targets(ticker)
    counts = _recommendation_counts(ticker)
    rec_mean = _compute_rec_mean(counts)
    rec_text = _compute_rec_text(rec_mean)

    if sleep_after:
        time.sleep(sleep_after)

    return {
        "ticker": ticker,
        "target_mean": targets.get("mean"),
        "target_high": targets.get("high"),
        "target_low": targets.get("low"),
        "rec_mean": rec_mean,
        "rec_text": rec_text,
        "strong_buy": counts.get("strong_buy"),
        "buy": counts.get("buy"),
        "hold": counts.get("hold"),
        "sell": counts.get("sell"),
        "strong_sell": counts.get("strong_sell"),
        "source": "finnhub.price-target + finnhub.stock/recommendation",
        "fetched_at_utc": now,
    }


def fetch_many(tickers: List[str]) -> List[Dict[str, Any]]:
    """Fetch broker snapshots for multiple tickers."""

    out: List[Dict[str, Any]] = []
    for t in tickers:
        try:
            out.append(fetch_broker_snapshot(t, sleep_after=0.5))
        except Exception as e:
            out.append(
                {
                    "ticker": t,
                    "error": str(e),
                    "source": "finnhub",
                    "fetched_at_utc": int(time.time()),
                }
            )
            time.sleep(0.6)
    return out

