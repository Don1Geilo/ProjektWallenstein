"""Generate price and sentiment overview text for tickers."""

from __future__ import annotations

from dataclasses import dataclass

import duckdb

from wallenstein.config import settings

from .db_utils import get_latest_prices
from .trending import fetch_weekly_returns

DB_PATH = settings.WALLENSTEIN_DB_PATH


@dataclass
class OverviewMessage:
    """Container for the compact and detailed Telegram messages."""

    compact: str
    detailed: str

    def as_text(self) -> str:
        parts = [self.compact.strip(), self.detailed.strip()]
        return "\n\n".join(part for part in parts if part)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.as_text()


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
) -> OverviewMessage:
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


    detail_lines: list[str] = ["📊 Wallenstein Markt-Update"]


    multi_hits: list[tuple[str, int]] = []
    multi_hit_symbols: list[str] = []
    if reddit_posts:
        for sym, posts in reddit_posts.items():
            if not posts:
                continue
            count = len(posts)
            if count >= 2:
                multi_hits.append((sym, count))
        multi_hits.sort(key=lambda x: x[1], reverse=True)
        multi_hit_symbols = [sym for sym, _ in multi_hits]

    with duckdb.connect(DB_PATH) as con:
        trending_rows: list[tuple[str, int, float | None, float | None]] = []
        try:
            trending_rows = con.execute(
                """
                SELECT ticker, mentions, avg_upvotes, hotness
                FROM reddit_trends
                WHERE date = CURRENT_DATE AND mentions >= 2
                ORDER BY hotness DESC, mentions DESC
                LIMIT 5
                """
            ).fetchall()
        except duckdb.Error:
            trending_rows = []

        weekly_map: dict[str, float] = {}
        weekly_targets: list[str] = []
        for sym in tickers:
            if sym not in weekly_targets:
                weekly_targets.append(sym)
        for ticker, *_ in trending_rows:
            if ticker not in weekly_targets:
                weekly_targets.append(ticker)

        for sym in multi_hit_symbols:
            if sym not in weekly_targets:
                weekly_targets.append(sym)

        if weekly_targets:
            try:
                weekly_map = fetch_weekly_returns(
                    con, weekly_targets, max_symbols=len(weekly_targets)
                )
            except Exception:
                weekly_map = {}

        # --- ML Signale (Buy/Sell) ---
        if trending_rows:
            detail_lines.append("")
            detail_lines.append("🔥 Reddit Trends:")
            for ticker, mentions, avg_up, hotness in trending_rows:
                avg_val = float(avg_up) if avg_up is not None else 0.0
                emoji = _hotness_to_emoji(hotness)
                suffix = f" {emoji}" if emoji else ""
                entry = f"- {ticker}: {int(mentions)} Mentions, AvgUp {avg_val:.1f}"
                weekly = weekly_map.get(str(ticker).upper()) if weekly_map else None
                if weekly is not None:
                    entry += f", 7d {weekly * 100:+.1f}%"
                entry += suffix
                detail_lines.append(entry)

        if multi_hits:
            detail_lines.append("")
            detail_lines.append("🔁 Mehrfach erwähnt:")
            for ticker, count in multi_hits:
                entry = f"- {ticker}: {count} Posts"
                weekly = weekly_map.get(str(ticker).upper()) if weekly_map else None
                if weekly is not None:
                    entry += f", 7d {weekly * 100:+.1f}%"
                detail_lines.append(entry)

        if trending_rows or multi_hits:
            detail_lines.append("")

        signal_rows: list[tuple[str, str, float | None, float | None, object, str | None]] = []
        try:
            signal_rows = con.execute(
                """
                WITH ranked AS (
                    SELECT *,
                           ROW_NUMBER() OVER (
                               PARTITION BY ticker, horizon_days
                               ORDER BY as_of DESC
                           ) AS rn
                    FROM predictions
                    WHERE horizon_days = 1
                )
                SELECT ticker, signal, confidence, expected_return, as_of, version
                FROM ranked
                WHERE rn = 1
                """,
            ).fetchall()
        except duckdb.Error:
            signal_rows = []

        latest_signals: dict[str, dict[str, object | None]] = {}
        for ticker, signal, confidence, expected_return, as_of, version in signal_rows:
            ticker_str = str(ticker).upper()
            latest_signals[ticker_str] = {
                "signal": str(signal).upper() if signal is not None else None,
                "confidence": float(confidence) if confidence is not None else None,
                "expected_return": float(expected_return)
                if expected_return is not None
                else None,
                "as_of": as_of,
                "version": str(version) if version is not None else None,
            }

        buy_rows = [row for row in signal_rows if str(row[1]).lower() == "buy"]
        sell_rows = [row for row in signal_rows if str(row[1]).lower() == "sell"]

        buy_rows.sort(key=lambda r: (float(r[2]) if r[2] is not None else 0.0), reverse=True)
        sell_rows.sort(key=lambda r: (float(r[2]) if r[2] is not None else 0.0), reverse=True)

        buy_rows = buy_rows[:5]
        sell_rows = sell_rows[:5]

        metrics_map: dict[str, dict[str, float | None]] = {}
        signal_symbols = [str(row[0]).upper() for row in buy_rows + sell_rows]
        if signal_symbols:
            placeholders = ",".join("?" for _ in signal_symbols)
            try:
                metric_rows = con.execute(
                    f"""
                    SELECT ticker, accuracy, f1, avg_strategy_return, long_win_rate
                    FROM model_training_state
                    WHERE ticker IN ({placeholders})
                    """,
                    signal_symbols,
                ).fetchall()
            except duckdb.Error:
                metric_rows = []
            for ticker_val, acc_val, f1_val, avg_ret, win_rate in metric_rows:
                metrics_map[str(ticker_val).upper()] = {
                    "accuracy": acc_val,
                    "f1": f1_val,
                    "avg_return": avg_ret,
                    "win_rate": win_rate,
                }

        if buy_rows or sell_rows:

            detail_lines.append("")
            detail_lines.append("🚦 ML Signale (1d Horizont):")

            if buy_rows:
                detail_lines.append("✅ Kauf:")

                for ticker, _signal, confidence, expected_return, as_of, version in buy_rows:
                    ticker_str = str(ticker).upper()
                    parts = []
                    if confidence is not None:
                        parts.append(f"{float(confidence) * 100:.1f}% Conviction")
                    if expected_return is not None:
                        parts.append(f"Erwartung {float(expected_return) * 100:+.2f}%")
                    metrics = metrics_map.get(ticker_str, {})
                    avg_ret = metrics.get("avg_return") if metrics else None
                    if avg_ret is not None:
                        parts.append(f"Backtest Ø {avg_ret * 100:+.2f}%")
                    win_rate = metrics.get("win_rate") if metrics else None
                    if win_rate is not None:
                        parts.append(f"Trefferquote {win_rate * 100:.1f}%")
                    if as_of and hasattr(as_of, "date"):
                        try:
                            parts.append(f"Stand {as_of.date()}")
                        except Exception:
                            pass
                    if version:
                        parts.append(str(version))

                    detail_lines.append("- " + ticker_str + ": " + ", ".join(parts))

            if sell_rows:
                detail_lines.append("⛔ Verkauf:")

                for ticker, _signal, confidence, expected_return, as_of, version in sell_rows:
                    ticker_str = str(ticker).upper()
                    parts = []
                    if confidence is not None:
                        parts.append(f"{float(confidence) * 100:.1f}% Conviction")
                    if expected_return is not None:
                        parts.append(f"Erwartung {float(expected_return) * 100:+.2f}%")
                    metrics = metrics_map.get(ticker_str, {})
                    avg_ret = metrics.get("avg_return") if metrics else None
                    if avg_ret is not None:
                        parts.append(f"Backtest Ø {avg_ret * 100:+.2f}%")
                    win_rate = metrics.get("win_rate") if metrics else None
                    if win_rate is not None:
                        parts.append(f"Trefferquote {win_rate * 100:.1f}%")
                    if as_of and hasattr(as_of, "date"):
                        try:
                            parts.append(f"Stand {as_of.date()}")
                        except Exception:
                            pass
                    if version:
                        parts.append(str(version))

                    detail_lines.append("- " + ticker_str + ": " + ", ".join(parts))

            detail_lines.append("")

        # Preis- und Change-Daten vorbereiten
        price_source = None
        for candidate in ("stocks", "stocks_view", "prices"):
            try:
                con.execute(f"SELECT 1 FROM {candidate} LIMIT 1")
            except duckdb.Error:
                continue
            price_source = candidate
            break

        summary_targets = list(dict.fromkeys([str(sym).upper() for sym in tickers]))
        for sym in signal_symbols:
            sym_u = str(sym).upper()
            if sym_u not in summary_targets:
                summary_targets.append(sym_u)

        change_map: dict[str, tuple[float | None, float | None]] = {}
        if price_source and summary_targets:
            placeholders = ",".join("?" for _ in summary_targets)
            try:
                change_rows = con.execute(
                    f"""
                    WITH ranked AS (
                        SELECT ticker, close, date,
                               ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) rn
                        FROM {price_source}
                        WHERE ticker IN ({placeholders})
                    )
                    SELECT ticker,
                           MAX(CASE WHEN rn = 1 THEN close END) AS last_close,
                           MAX(CASE WHEN rn = 2 THEN close END) AS prev_close
                    FROM ranked
                    GROUP BY ticker
                    """,
                    summary_targets,
                ).fetchall()
            except duckdb.Error:
                change_rows = []
            for ticker, last_close, prev_close in change_rows:
                change_map[str(ticker).upper()] = (
                    float(last_close) if last_close is not None else None,
                    float(prev_close) if prev_close is not None else None,
                )

        def _format_change_pct(ticker_symbol: str) -> float | None:
            change_tuple = change_map.get(ticker_symbol.upper())
            if not change_tuple:
                return None
            last_close, prev_close = change_tuple
            if last_close is None or prev_close in (None, 0):
                return None
            return (last_close / prev_close - 1) * 100

        compact_lines: list[str] = []

        def _format_buy_summary() -> str:
            if not buy_rows:
                return "Top Kauf-Signale: aktuell keine frischen Empfehlungen"
            summaries: list[str] = []
            for ticker, _signal, confidence, expected_return, _as_of, _version in buy_rows[:3]:
                ticker_str = str(ticker).upper()
                price = prices_usd.get(ticker_str)
                change_pct = _format_change_pct(ticker_str)
                parts: list[str] = []
                if price is not None:
                    parts.append(f"{price:.2f} USD")
                if change_pct is not None:
                    parts.append(f"1d {change_pct:+.1f}%")
                if expected_return is not None:
                    parts.append(f"Ziel {float(expected_return) * 100:+.1f}%")
                if confidence is not None:
                    parts.append(f"Conv. {float(confidence) * 100:.0f}%")
                summary = ticker_str
                if parts:
                    summary += " (" + ", ".join(parts) + ")"
                summaries.append(summary)
            return "Top Kauf-Signale: " + "; ".join(summaries)

        def _format_sell_summary() -> str | None:
            if not sell_rows:
                return None
            snippets: list[str] = []
            for ticker, _signal, confidence, expected_return, _as_of, _version in sell_rows[:2]:
                ticker_str = str(ticker).upper()
                change_pct = _format_change_pct(ticker_str)
                parts: list[str] = []
                if change_pct is not None:
                    parts.append(f"1d {change_pct:+.1f}%")
                if expected_return is not None:
                    parts.append(f"Ziel {float(expected_return) * 100:+.1f}%")
                if confidence is not None:
                    parts.append(f"Conv. {float(confidence) * 100:.0f}%")
                snippet = ticker_str
                if parts:
                    snippet += " (" + ", ".join(parts) + ")"
                snippets.append(snippet)
            if snippets:
                return "Achtung Verkauf: " + "; ".join(snippets)
            return None

        def _format_trend_summary() -> str | None:
            if not trending_rows:
                return None
            ticker, mentions, avg_up, _hotness = trending_rows[0]
            weekly = weekly_map.get(str(ticker).upper()) if weekly_map else None
            parts = [f"{int(mentions)} Mentions"]
            if avg_up is not None:
                parts.append(f"AvgUp {float(avg_up):.0f}")
            if weekly is not None:
                parts.append(f"7d {weekly * 100:+.1f}%")
            return f"Heißeste Diskussion: {ticker} (" + ", ".join(parts) + ")"

        compact_lines.append("⚡️ Schnellüberblick")
        compact_lines.append(_format_buy_summary())
        sell_summary = _format_sell_summary()
        if sell_summary:
            compact_lines.append(sell_summary)
        trend_summary = _format_trend_summary()
        if trend_summary:
            compact_lines.append(trend_summary)
        for t in tickers:
            ticker_upper = t.upper()
            usd = prices_usd.get(t)
            eur = prices_eur.get(t)
            detail_lines.append(f"📈 {t}")

            try:
                alias_rows = con.execute(
                    "SELECT alias FROM ticker_aliases WHERE ticker = ? ORDER BY alias",
                    [t],
                ).fetchall()
            except duckdb.Error:
                alias_rows = []
            aliases = ", ".join(a for a, in alias_rows if a)
            if aliases:
                detail_lines.append(f"Alias: {aliases}")

            change_tuple = change_map.get(ticker_upper)
            change_pct: float | None = None
            if change_tuple:
                last_close, prev_close = change_tuple
                if last_close is not None and prev_close not in (None, 0):
                    change_pct = (last_close / prev_close - 1) * 100

            price_bits: list[str] = []
            if usd is not None:
                price_bits.append(f"{usd:.2f} USD")
            if eur is not None:
                price_bits.append(f"{eur:.2f} EUR")
            if change_pct is not None:
                price_bits.append(f"1d {change_pct:+.2f}%")
            detail_lines.append("Preis: " + (" | ".join(price_bits) if price_bits else "n/a"))

            weekly = weekly_map.get(ticker_upper) if weekly_map else None
            if weekly is not None:
                detail_lines.append(f"Trend 7d: {weekly * 100:+.1f}%")

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
            detail_lines.append(f"Sentiment 7d: {w_sent:+.2f}")

            try:
                sent_row_1d = con.execute(
                    "SELECT sentiment_weighted FROM reddit_sentiment_daily WHERE ticker=? ORDER BY date DESC LIMIT 1",
                    [t],
                ).fetchone()
            except duckdb.Error:
                sent_row_1d = None
            if sent_row_1d and sent_row_1d[0] is not None:
                detail_lines.append(f"Sentiment 1d: {sent_row_1d[0]:+.2f}")

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
                detail_lines.append(f"Sentiment 24h: {sent_row_24h[0]:+.2f}")

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
            detail_lines.append(f"Mentions: {mentions} ({change:+.0f}% ggü. 7d Ø)")
            emoji = _hotness_to_emoji(hotness)
            if emoji:
                detail_lines.append(f"Hotness: {emoji}")

            ticker_signal = latest_signals.get(ticker_upper)
            if ticker_signal and ticker_signal.get("signal"):
                sig = str(ticker_signal["signal"]).upper()
                parts: list[str] = [sig]
                conf_val = ticker_signal.get("confidence")
                if isinstance(conf_val, (int, float)):
                    parts.append(f"{float(conf_val) * 100:.1f}% Conviction")
                exp_val = ticker_signal.get("expected_return")
                if isinstance(exp_val, (int, float)):
                    parts.append(f"Erwartung {float(exp_val) * 100:+.2f}%")
                version_val = ticker_signal.get("version")
                if version_val:
                    parts.append(str(version_val))
                detail_lines.append("Signal: " + ", ".join(parts))

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
                detail_lines.append(f"Kurs seit Top-Post (+3d): {top_post[0] * 100:+.1f}%")

            detail_lines.append("")


    compact_text = "\n".join(compact_lines).strip()
    detail_text = "\n".join(detail_lines).strip()
    return OverviewMessage(compact=compact_text, detailed=detail_text)
