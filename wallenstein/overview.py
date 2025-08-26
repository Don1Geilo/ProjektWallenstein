"""Generate price and sentiment overview text for tickers."""

from __future__ import annotations

from wallenstein.config import settings

from .db_utils import get_latest_prices
from .sentiment import analyze_sentiment_batch, derive_recommendation

DB_PATH = settings.WALLENSTEIN_DB_PATH


def _fetch_latest_price(ticker: str) -> float | None:
    """Fetch the latest USD price for ``ticker`` via yfinance."""
    try:
        import yfinance as yf

        tk = yf.Ticker(ticker)
        price = getattr(getattr(tk, "fast_info", None), "last_price", None)
        if price is None:
            hist = tk.history(period="1d", interval="1d", auto_adjust=False, actions=False)
            if hist is not None and not hist.empty:
                price = float(hist["Close"].iloc[-1])
        return float(price) if price is not None else None
    except Exception:  # pragma: no cover - network issues
        return None


def _fetch_usd_per_eur_rate() -> float | None:
    """Return USD per 1 EUR via yfinance."""
    try:
        import yfinance as yf

        tk = yf.Ticker("EURUSD=X")
        rate = getattr(getattr(tk, "fast_info", None), "last_price", None)
        if rate is None:
            hist = tk.history(period="1d", interval="1d", auto_adjust=False, actions=False)
            if hist is not None and not hist.empty:
                rate = float(hist["Close"].iloc[-1])
        return float(rate) if rate is not None else None
    except Exception:  # pragma: no cover - network issues
        return None


def generate_overview(
    tickers: list[str],
    reddit_posts: dict[str, list[dict]] | None = None,
) -> str:
    """Return a formatted overview for ``tickers``.

    The overview lists the latest USD and EUR prices and a simple Reddit-based
    sentiment score with a derived recommendation for each ticker. Pass in
    ``reddit_posts`` to reuse already fetched Reddit data; otherwise an empty
    mapping is assumed and no fetch is performed.
    """

    prices_usd = get_latest_prices(DB_PATH, tickers, use_eur=False)
    prices_eur = get_latest_prices(DB_PATH, tickers, use_eur=True)
    # Fallback: fetch missing USD prices via yfinance and convert to EUR
    missing = [t for t in tickers if prices_usd.get(t) is None]
    usd_per_eur = _fetch_usd_per_eur_rate() if missing else None
    for t in missing:
        px = _fetch_latest_price(t)
        if px is not None:
            prices_usd[t] = px
            if usd_per_eur:
                prices_eur[t] = px / usd_per_eur

    reddit_posts = reddit_posts or {t: [] for t in tickers}

    sentiments = {}
    for ticker in tickers:
        posts = reddit_posts.get(ticker, [])
        texts = [p.get("text", "") for p in posts if p.get("text")]
        if texts:
            scores = analyze_sentiment_batch(texts)
            sentiments[ticker] = sum(scores) / len(scores)
        else:
            sentiments[ticker] = 0.0

    price_lines = []
    sentiment_lines = []
    for t in tickers:
        usd = prices_usd.get(t)
        eur = prices_eur.get(t)
        if usd is not None and eur is not None:
            price_lines.append(f"{t}: {usd:.2f} USD ({eur:.2f} EUR)")
        elif usd is not None:
            price_lines.append(f"{t}: {usd:.2f} USD")
        elif eur is not None:
            price_lines.append(f"{t}: {eur:.2f} EUR")
        else:
            price_lines.append(f"{t}: n/a")

        sent = sentiments.get(t, 0.0)
        rec = derive_recommendation(sent)
        sentiment_lines.append(f"{t}: Sentiment {sent:+.2f} | {rec}")

    return "📊 Wallenstein Übersicht\n" + "\n".join(price_lines + sentiment_lines)
