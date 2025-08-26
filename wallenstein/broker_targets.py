# wallenstein/broker_targets.py
from __future__ import annotations

"""Fetch analyst price targets and recommendations using the FMP API."""

import logging
import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from wallenstein.config import settings

log = logging.getLogger("wallenstein.targets")

# FMP configuration
FMP_API_KEY = settings.FMP_API_KEY
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
def _sf(v: Any) -> float | None:
    """Convert value to float if possible."""

    try:
        return float(v) if v is not None and str(v).strip() != "" else None
    except Exception:
        return None


def _pos(x: float | None) -> float | None:
    """Return value only if positive."""

    return x if (x is not None and x > 0) else None


def _fmp_get(path: str, params: dict[str, Any]) -> dict[str, Any]:
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
def _parse_price_target_item(item: dict[str, Any]) -> dict[str, float | None]:
    """Map raw FMP price target item to our internal structure.

    The FMP API exposes slightly different field names depending on the
    endpoint or plan. We therefore try a couple of alternatives for each
    value so the function works with both price-target and
    price-target-consensus responses.
    """

    def _first(*keys: str) -> Any:
        for k in keys:
            if k in item:
                return item[k]
        return None

    mean_keys = (
        "targetConsensus",
        "priceTargetAverage",
        "priceTarget",
        "targetPrice",
        "targetMean",
    )

    return {
        "target_mean": _pos(_sf(_first(*mean_keys))),
        "target_high": _pos(_sf(_first("targetHigh", "priceTargetHigh", "targetPriceHigh"))),
        "target_low": _pos(_sf(_first("targetLow", "priceTargetLow", "targetPriceLow"))),
        "target_median": _pos(
            _sf(_first("targetMedian", "priceTargetMedian", "targetPriceMedian"))
        ),
    }


def _fmp_price_target(ticker: str) -> dict[str, Any]:
    """Return price targets and recommendation counts for ``ticker``.

    On certain errors (e.g. 403) a dictionary with an ``error`` key is
    returned so callers can handle it explicitly.
    """

    out: dict[str, float | None] = {
        "target_mean": None,
        "target_high": None,
        "target_low": None,
        "target_median": None,
        "strong_buy": None,
        "buy": None,
        "hold": None,
        "sell": None,
        "strong_sell": None,
    }

    data = _fmp_get("price-target-consensus", {"symbol": ticker})
    fallback_used = False
    if (isinstance(data, dict) and data.get("error")) or not data:
        msg = data.get("error") if isinstance(data, dict) else "empty response"
        log.warning(f"[{ticker}] price-target-consensus {msg}; falling back to price-target")
        data = _fmp_get("price-target", {"symbol": ticker})
        fallback_used = True

    if isinstance(data, dict) and data.get("error"):
        if fallback_used:
            log.warning(f"[{ticker}] price-target fallback failed: {data['error']}")
        return data

    try:
        if isinstance(data, list) and data:
            item = data[0]
        elif isinstance(data, dict):
            item = data
        else:
            item = {}
        out.update(_parse_price_target_item(item))
        out.update(
            {
                "strong_buy": _sf(item.get("strongBuy")),
                "buy": _sf(item.get("buy")),
                "hold": _sf(item.get("hold")),
                "sell": _sf(item.get("sell")),
                "strong_sell": _sf(item.get("strongSell")),
            }
        )
    except Exception as e:  # pragma: no cover - defensive
        log.warning(f"[{ticker}] price-target error: {e}")

    return out


def _fmp_price_targets(tickers: list[str]) -> dict[str, dict[str, Any]]:
    """Fetch price targets for multiple tickers with a single API call."""
    data = _fmp_get("price-target-consensus", {"symbol": ",".join(tickers)})
    if (isinstance(data, dict) and data.get("error")) or not data:
        msg = data.get("error") if isinstance(data, dict) else "empty response"
        log.warning(f"price-target-consensus bulk {msg}; falling back to per-ticker requests")
        return {t: _fmp_price_target(t) for t in tickers}
    if isinstance(data, dict):
        data = [data]

    out: dict[str, dict[str, Any]] = {}
    for item in data or []:
        t = str(item.get("symbol") or item.get("ticker") or "").upper()
        out[t] = _parse_price_target_item(item)

    # Ensure every requested ticker has an entry
    for t in tickers:
        out.setdefault(
            t,
            {
                "target_mean": None,
                "target_high": None,
                "target_low": None,
                "target_median": None,
            },
        )
    return out


def _compute_rec_mean(counts: dict[str, int | None]) -> float | None:
    """Calculate a mean recommendation score from analyst counts."""

    weights = {"strong_buy": 1, "buy": 2, "hold": 3, "sell": 4, "strong_sell": 5}
    total = sum(v for v in counts.values() if isinstance(v, (int, float)))
    if not total:
        return None
    score = sum(weights[k] * (counts.get(k) or 0) for k in weights)
    return score / total


def _compute_rec_text(mean: float | None) -> str | None:
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
def fetch_broker_snapshot(ticker: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fetch a broker snapshot for a single ``ticker`` using FMP.

    If ``data`` is provided it must contain the already fetched price target
    information for ``ticker`` and no additional HTTP request will be made.
    """

    now = int(time.time())
    if data is None:
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


def fetch_many(tickers: list[str], *, sleep_between: float = 0.0) -> list[dict[str, Any]]:
    """Fetch broker snapshots for multiple tickers with a single API call."""

    out: list[dict[str, Any]] = []
    try:
        data_map = _fmp_price_targets(tickers)
    except Exception as e:  # pragma: no cover - defensive
        now = int(time.time())
        err = str(e)
        for t in tickers:
            out.append(
                {
                    "ticker": t,
                    "error": err,
                    "source": "fmp.price-target-consensus",
                    "fetched_at_utc": now,
                }
            )
        return out
    for t in tickers:
        try:
            out.append(fetch_broker_snapshot(t, data=data_map.get(t, {})))
        except Exception as e:  # pragma: no cover - defensive
            out.append(
                {
                    "ticker": t,
                    "error": str(e),
                    "source": "fmp.price-target-consensus",
                    "fetched_at_utc": int(time.time()),
                }
            )
        if sleep_between:
            time.sleep(sleep_between)
    return out
