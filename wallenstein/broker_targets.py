# wallenstein/broker_targets.py
from __future__ import annotations

"""Fetch analyst price targets and recommendations using the FMP API."""

import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger("wallenstein.targets")

# FMP configuration
FMP_API_KEY = os.getenv("FMP_API_KEY", "demo")
FMP_BASE_URL = "https://financialmodelingprep.com/api/v4"


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


def _fmp_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Perform a GET request against the FMP API and return JSON."""

    params = dict(params)
    params["apikey"] = FMP_API_KEY
    url = f"{FMP_BASE_URL}/{path}"
    try:
        r = _SESSION.get(url, params=params, timeout=10.0)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as exc:  # pragma: no cover - network
        status = exc.response.status_code if exc.response else None
        if status == 403:
            msg = (
                "FMP API responded with 403 Forbidden. "
                "Your API key might be invalid, expired or lacks permission."
            )
            return {"error": {"status": 403, "message": msg, "url": url}}
        raise


# ---------------------------------------------------------------------------
# FMP specific helpers
# ---------------------------------------------------------------------------
def _fmp_price_target(ticker: str) -> Dict[str, Any]:
    """Return price targets and recommendation counts for ``ticker``.

    On certain errors (e.g. 403) a dictionary with an ``error`` key is
    returned so callers can handle it explicitly.
    """

    out: Dict[str, Optional[float]] = {
        "target_mean": None,
        "target_high": None,
        "target_low": None,
        "strong_buy": None,
        "buy": None,
        "hold": None,
        "sell": None,
        "strong_sell": None,
    }

    data = _fmp_get("price-target-consensus", {"symbol": ticker})
    if isinstance(data, dict) and data.get("error"):
        return data

    try:
        item = data[0] if isinstance(data, list) and data else {}
        out["target_mean"] = _pos(_sf(item.get("targetConsensus")))
        out["target_high"] = _pos(_sf(item.get("targetHigh")))
        out["target_low"] = _pos(_sf(item.get("targetLow")))
    except Exception as e:
        log.warning(f"[{ticker}] price-target error: {e}")
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
def fetch_broker_snapshot(ticker: str) -> Dict[str, Any]:
    """Fetch a broker snapshot for a single ``ticker`` using FMP."""

    now = int(time.time())
    data = _fmp_price_target(ticker)

    if isinstance(data, dict) and data.get("error"):
        return {
            "ticker": ticker,
            "error": data["error"],
            "source": "fmp.price-target-consensus",
            "fetched_at_utc": now,
        }

    counts = {
        "strong_buy": data.get("strong_buy"),
        "buy": data.get("buy"),
        "hold": data.get("hold"),
        "sell": data.get("sell"),
        "strong_sell": data.get("strong_sell"),
    }
    rec_mean = _compute_rec_mean(counts)
    rec_text = _compute_rec_text(rec_mean)

    return {
        "ticker": ticker,
        "target_mean": data.get("target_mean"),
        "target_high": data.get("target_high"),
        "target_low": data.get("target_low"),
        "rec_mean": rec_mean,
        "rec_text": rec_text,
        "strong_buy": counts.get("strong_buy"),
        "buy": counts.get("buy"),
        "hold": counts.get("hold"),
        "sell": counts.get("sell"),
        "strong_sell": counts.get("strong_sell"),
        "source": "fmp.price-target-consensus",
        "fetched_at_utc": now,
    }


def fetch_many(tickers: List[str], *, sleep_between: float = 0.25) -> List[Dict[str, Any]]:
    """Fetch broker snapshots for multiple tickers."""

    out: List[Dict[str, Any]] = []
    for t in tickers:
        try:
            out.append(fetch_broker_snapshot(t))
        except Exception as e:
            out.append(
                {
                    "ticker": t,
                    "error": str(e),
                    "source": "fmp",
                    "fetched_at_utc": int(time.time()),
                }
            )
        if sleep_between:
            time.sleep(sleep_between)
    return out

