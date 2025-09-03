"""Generate price and sentiment overview text for tickers."""

from __future__ import annotations

import duckdb

from wallenstein.config import settings

from .db_utils import get_latest_prices

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

    def _hotness_to_emoji(val: float | None) -> str:
        if val is None or val <= 0:
            return ""
        count = min(int(val // 500), 3)
        return "🔥" * count

    lines = ["📊 Wallenstein Übersicht"]
    with duckdb.connect(DB_PATH) as con:
        for t in tickers:
            usd = prices_usd.get(t)
            eur = prices_eur.get(t)
            lines.append(f"📈 {t}")
            if usd is not None and eur is not None:
                lines.append(f"{t}: {usd:.2f} USD ({eur:.2f} EUR)")
            elif usd is not None:
                lines.append(f"{t}: {usd:.2f} USD")
            elif eur is not None:
                lines.append(f"{t}: {eur:.2f} EUR")
            else:
                lines.append(f"{t}: n/a")

            try:
                sent_row = con.execute(
                    """
                    SELECT AVG(sentiment_dict) AS s, AVG(sentiment_weighted) AS w
                    FROM reddit_sentiment_daily
                    WHERE ticker = ? AND date >= CURRENT_DATE - INTERVAL 7 DAY
                    """,
                    [t],
                ).fetchone()
            except duckdb.Error:
                sent_row = None
            w_sent = sent_row[1] if sent_row and sent_row[1] is not None else 0.0
            lines.append(f"Sentiment (7d, weighted): {w_sent:+.2f}")

            try:
                sent_row_1d = con.execute(
                    "SELECT sentiment_weighted FROM reddit_sentiment_daily WHERE ticker=? ORDER BY date DESC LIMIT 1",
                    [t],
                ).fetchone()
            except duckdb.Error:
                sent_row_1d = None
            if sent_row_1d and sent_row_1d[0] is not None:
                lines.append(f"Sentiment (1d, weighted): {sent_row_1d[0]:+.2f}")

            try:
                sent_row_24h = con.execute(
                    """
                    SELECT AVG(sentiment_weighted) AS w
                    FROM reddit_sentiment_hourly
                    WHERE ticker = ? AND created_utc >= NOW() - INTERVAL 24 HOUR
                    """,
                    [t],
                ).fetchone()
            except duckdb.Error:
                sent_row_24h = None
            if sent_row_24h and sent_row_24h[0] is not None:
                lines.append(f"Sentiment (24h, weighted): {sent_row_24h[0]:+.2f}")

            try:
                trend_today = con.execute(
                    "SELECT mentions, hotness FROM reddit_trends WHERE date = CURRENT_DATE AND ticker = ?",
                    [t],
                ).fetchone()
            except duckdb.Error:
                trend_today = None
            mentions = trend_today[0] if trend_today else 0
            hotness = trend_today[1] if trend_today else 0
            try:
                trend_avg = con.execute(
                    """
                    SELECT AVG(mentions) FROM reddit_trends
                    WHERE date >= CURRENT_DATE - INTERVAL 7 DAY
                      AND date < CURRENT_DATE AND ticker = ?
                    """,
                    [t],
                ).fetchone()
            except duckdb.Error:
                trend_avg = None
            avg_mentions = trend_avg[0] if trend_avg and trend_avg[0] else 0
            change = ((mentions / avg_mentions) - 1) * 100 if avg_mentions else 0.0
            lines.append(f"Mentions: {mentions} ({change:+.0f}% ggü. 7d Ø)")
            emoji = _hotness_to_emoji(hotness)
            if emoji:
                lines.append(f"Hotness: {emoji}")

            try:
                top_post = con.execute(
                    """
                    SELECT return_3d FROM reddit_enriched
                    WHERE ticker = ? AND sentiment_weighted IS NOT NULL AND return_3d IS NOT NULL
                    ORDER BY sentiment_weighted DESC LIMIT 1
                    """,
                    [t],
                ).fetchone()
            except duckdb.Error:
                top_post = None
            if top_post and top_post[0] is not None:
                lines.append(f"Kurs seit Top-Post (+3d): {top_post[0] * 100:+.1f}%")

            lines.append("")

    return "\n".join(lines).strip()
