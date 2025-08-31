import argparse
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv

# --- .env laden ---
env_loaded = load_dotenv(find_dotenv(usecwd=True), override=True)
if not env_loaded:
    alt_path = Path(__file__).with_name(".env")
    load_dotenv(dotenv_path=alt_path, override=True)

from wallenstein.config import settings, validate_config
from wallenstein.db import init_schema

validate_config()

# --- Logging ---
LOG_LEVEL = settings.LOG_LEVEL.upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("wallenstein")

try:
    from wallenstein.notify import notify_telegram
except Exception as e:  # pragma: no cover
    log.warning(f"notify_telegram nicht verfügbar: {e}")

    def notify_telegram(text: str) -> bool:
        log.warning("Telegram nicht konfiguriert – Nachricht nicht gesendet.")
        return False


try:
    from wallenstein.reddit_scraper import update_reddit_data
except Exception as e:  # pragma: no cover
    log.warning(f"update_reddit_data nicht verfügbar: {e}")

    # Fallback: akzeptiert zusätzliche Keyword-Args wie include_comments
    def update_reddit_data(tickers, subreddits=None, limit_per_sub=50, **kwargs):
        return {t: [] for t in tickers}


# Alerts API optional
try:
    import alerts_api
except Exception as e:  # pragma: no cover
    log.warning(f"alerts_api nicht verfügbar: {e}")
    alerts_api = None

# --- Pfade/Konfig ---
DB_PATH = settings.WALLENSTEIN_DB_PATH
os.makedirs(Path(DB_PATH).parent, exist_ok=True)  # stellt sicher, dass data/ existiert
init_schema(DB_PATH)

# Telegram (zur Rückwärtskompatibilität beibehalten)
TELEGRAM_BOT_TOKEN = (settings.TELEGRAM_BOT_TOKEN or "").strip()
TELEGRAM_CHAT_ID = (settings.TELEGRAM_CHAT_ID or "").strip()

# --- Projekt-Module ---
from wallenstein.db_utils import ensure_prices_view, get_latest_prices
from wallenstein.models import train_per_stock
from wallenstein.overview import generate_overview
from wallenstein.reddit_enrich import (
    compute_reddit_sentiment,
    compute_reddit_trends,
    compute_returns,
    enrich_reddit_posts,
)
from wallenstein.sentiment import analyze_sentiment_batch
from wallenstein.sentiment_analysis import analyze_sentiment
from wallenstein.stock_data import purge_old_prices, update_fx_rates, update_prices
from wallenstein.trending import (
    auto_add_candidates_to_watchlist,
    scan_reddit_for_candidates,
)


def resolve_tickers(override: str | None = None) -> list[str]:
    """Return the list of ticker symbols to process.

    The function looks for a comma separated list of tickers provided via
    ``override`` first.  If no override is given, the symbols are read from the
    persisted watchlist stored in the DuckDB database.  All symbols are
    upper–cased and duplicates are removed by the underlying SQL query.
    """
    if override:
        tickers = [t.strip().upper() for t in override.split(",") if t.strip()]
        if not tickers:
            log.warning("Keine Ticker im --tickers Override gefunden")
        return tickers

    # Aus DuckDB lesen (chat-übergreifend, DISTINCT)
    try:
        from wallenstein.watchlist import all_unique_symbols as wl_all

        with duckdb.connect(DB_PATH) as con:
            tickers = [s.strip().upper() for s in wl_all(con)]
    except Exception as exc:  # pragma: no cover
        log.warning(f"Watchlist-Abfrage fehlgeschlagen: {exc}")
        tickers = []

    if not tickers:
        log.warning("Keine Ticker gefunden")
    return tickers


def train_model_for_ticker(
    ticker: str,
    db_path: str,
    sentiment_frames: dict[str, pd.DataFrame],
) -> tuple[str, float | None, float | None, float | None, float | None, float | None]:
    """Train the per-stock model for a single ticker."""
    try:
        with duckdb.connect(db_path) as con:
            df_price = con.execute(
                "SELECT date, close FROM prices WHERE ticker = ? ORDER BY date",
                [ticker],
            ).fetchdf()
        if df_price.empty:
            log.info(f"{ticker}: Keine Preisdaten – Training übersprungen")
            return ticker, None, None, None, None, None

        df_price["date"] = pd.to_datetime(df_price["date"]).dt.normalize()
        df_sent = sentiment_frames.get(ticker, pd.DataFrame(columns=["date", "sentiment"])).copy()
        if not df_sent.empty:
            df_sent["date"] = pd.to_datetime(df_sent["date"]).dt.normalize()

        df_stock = pd.merge(df_price, df_sent, on="date", how="left")
        acc, f1, roc_auc, precision, recall = train_per_stock(df_stock)
        if acc is None:
            log.info(f"{ticker}: Zu wenige Daten für Modelltraining")
        return ticker, acc, f1, roc_auc, precision, recall
    except Exception as e:  # pragma: no cover
        log.warning(f"{ticker}: Modelltraining fehlgeschlagen: {e}")
        return ticker, None, None, None, None, None


# --- Sentiment aggregation helpers ---


def aggregate_daily_sentiment(posts: pd.DataFrame) -> pd.DataFrame:
    if posts.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "n_posts",
                "sentiment_mean",
                "sentiment_weighted",
                "sentiment_median",
                "updated_at",
            ]
        )
    posts = posts.copy()
    posts["text"] = posts["title"].fillna("") + " " + posts["selftext"].fillna("")
    posts["sentiment"] = posts["text"].map(analyze_sentiment)
    posts["ups"] = pd.to_numeric(posts["ups"], errors="coerce").fillna(0).astype(int)
    posts["num_comments"] = (
        pd.to_numeric(posts["num_comments"], errors="coerce").fillna(0).astype(int)
    )
    posts["weight"] = (
        1
        + np.log10(1 + posts["ups"])
        + 0.2 * np.log10(1 + posts["num_comments"])
    )
    posts["date"] = pd.to_datetime(posts["created_utc"], unit="s").dt.tz_localize("UTC").dt.date
    posts["sentiment_weight"] = posts["sentiment"] * posts["weight"]
    agg = (
        posts.groupby(["date", "ticker"])
        .agg(
            n_posts=("sentiment", "size"),
            sentiment_mean=("sentiment", "mean"),
            sent_w_sum=("sentiment_weight", "sum"),
            weight_sum=("weight", "sum"),
            sentiment_median=("sentiment", "median"),
        )
        .reset_index()
    )
    agg["sentiment_weighted"] = agg["sent_w_sum"] / agg["weight_sum"].clip(lower=1e-9)
    agg = agg.drop(columns=["sent_w_sum", "weight_sum"])
    agg["updated_at"] = datetime.now(timezone.utc)
    agg = agg[
        [
            "date",
            "ticker",
            "n_posts",
            "sentiment_mean",
            "sentiment_weighted",
            "sentiment_median",
            "updated_at",
        ]
    ]
    return agg


def upsert_reddit_daily_sentiment(con, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS reddit_daily_sentiment(
            date DATE,
            ticker TEXT,
            n_posts INTEGER,
            sentiment_mean DOUBLE,
            sentiment_weighted DOUBLE,
            sentiment_median DOUBLE,
            updated_at TIMESTAMP
        )
        """
    )
    try:
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_rds_ticker_date ON reddit_daily_sentiment(ticker, date)"
        )
    except Exception:
        pass
    con.register("df", df)
    con.execute(
        "DELETE FROM reddit_daily_sentiment WHERE (date, ticker) IN (SELECT date, ticker FROM df)"
    )
    con.execute("INSERT INTO reddit_daily_sentiment SELECT * FROM df")
    return len(df)


# ---------- Main ----------
def run_pipeline(tickers: list[str] | None = None) -> int:
    """Execute the complete Wallenstein data pipeline."""
    t0 = time.time()
    log.info("🚀 Start Wallenstein: Pipeline-Run")

    if tickers is None:
        tickers = resolve_tickers()
    if not tickers:
        log.warning("Keine Ticker zur Verarbeitung – Pipeline abgebrochen")
        return 1

    reddit_posts = {t: [] for t in tickers}
    added = 0
    fx_added = 0

    # Parallel: Preise, Reddit, FX
    with ThreadPoolExecutor(max_workers=settings.PIPELINE_MAX_WORKERS) as executor:
        fut_prices = executor.submit(update_prices, DB_PATH, tickers)
        fut_reddit = executor.submit(
            update_reddit_data,
            tickers,
            ["wallstreetbets", "wallstreetbetsGer", "mauerstrassenwetten"],
            include_comments=True,
        )
        fut_fx = executor.submit(update_fx_rates, DB_PATH)

        try:
            added = fut_prices.result()
            if added:
                log.info(f"✅ Kursdaten aktualisiert: +{added} neue Zeilen")
        except Exception as e:
            log.error(f"❌ Kursupdate fehlgeschlagen: {e}")
        purge_old_prices(DB_PATH)

        try:
            reddit_posts = fut_reddit.result()
            log.info("✅ Reddit-Daten aktualisiert")
        except Exception as e:
            log.error(f"❌ Reddit-Update fehlgeschlagen: {e}")
            reddit_posts = {t: [] for t in tickers}

        try:
            fx_added = fut_fx.result()
            log.info(f"FX-Update: +{fx_added} neue EURUSD-Zeilen")
        except Exception as e:
            log.error(f"❌ FX-Update fehlgeschlagen: {e}")

    # Sentiment aggregation and persistence
    try:
        rows_list: list[dict] = []
        for tkr, posts in reddit_posts.items():
            for p in posts:
                created = p.get("created_utc")
                if hasattr(created, "timestamp"):
                    created_val = int(created.timestamp())
                else:
                    created_val = created
                rows_list.append(
                    {
                        "id": p.get("id"),
                        "created_utc": created_val,
                        "ticker": tkr,
                        "title": p.get("title", ""),
                        "selftext": p.get("text", ""),
                        "ups": p.get("upvotes", 0),
                        "num_comments": p.get("num_comments", 0),
                    }
                )
        posts_df = pd.DataFrame(rows_list)
        with duckdb.connect(DB_PATH) as con:
            agg = aggregate_daily_sentiment(posts_df)
            rows = upsert_reddit_daily_sentiment(con, agg)
        log.info(
            f"✅ Sentiment: {rows} rows upserted across {agg['ticker'].nunique() if not agg.empty else 0} tickers"
        )
    except Exception as e:
        log.error(f"❌ Sentiment-Aggregation fehlgeschlagen: {e}")

    # Enrichment + Trends + Returns
    try:
        with duckdb.connect(DB_PATH) as con:
            enriched = enrich_reddit_posts(con, reddit_posts)
            log.info(f"Reddit-Enrichment: +{enriched} Zeilen")
            trends = compute_reddit_trends(con)
            log.info(f"Trends aktualisiert: {trends}")
            returns = compute_returns(con)
            log.info(f"Returns berechnet: {returns} Posts")

            # Sentiment-Aggregate (hourly/daily) direkt nachziehen
            h_count, d_count = compute_reddit_sentiment(con, backfill_days=14)
            log.info(f"Sentiment-Aggregate aktualisiert: hourly≈{h_count}, daily≈{d_count}")
    except Exception as e:
        log.error(f"❌ Reddit-Enrichment/Sentiments fehlgeschlagen: {e}")
    try:
        with duckdb.connect(DB_PATH) as con:
            cands = scan_reddit_for_candidates(
                con,
                lookback_days=7,
                window_hours=24,
                min_mentions=20,
                min_lift=3.0,
            )
            if cands:
                top_preview = ", ".join(
                    [f"{c.symbol} (m24h={c.mentions_24h}, x{c.lift:.1f})" for c in cands[:5]]
                )
                log.info(f"Trending-Kandidaten (Top 5): {top_preview}")
            else:
                log.info("Trending-Kandidaten: keine")

            added_syms = auto_add_candidates_to_watchlist(
                con,
                notify_fn=notify_telegram,
                max_new=3,
                min_mentions=30,
                min_lift=4.0,
            )
            if added_syms:
                log.info(f"Auto zur Watchlist hinzugefügt: {', '.join(added_syms)}")
    except Exception as e:
        log.warning(f"Trending-Scan fehlgeschlagen: {e}")

    # View sicherstellen & Preise ziehen
    ensure_prices_view(DB_PATH, view_name="stocks", table_name="prices")
    prices_usd = get_latest_prices(DB_PATH, tickers, use_eur=False)
    log.info(f"USD: {prices_usd}")
    if alerts_api:
        try:
            alerts_api.active_alerts(prices_usd, notify_telegram)
        except Exception as e:  # pragma: no cover
            log.warning(f"Alertprüfung fehlgeschlagen: {e}")

    # Sentiment je Ticker aus Rohposts (für Modelle)
    sentiments: dict[str, float] = {}
    sentiment_frames: dict[str, pd.DataFrame] = {}

    for ticker, texts in reddit_posts.items():
        texts = list(texts) if texts is not None else []
        if texts:
            df_posts = pd.DataFrame(texts)

            # Erwartete Spalte: "text"
            if "text" in df_posts:
                df_posts["sentiment"] = analyze_sentiment_batch(
                    df_posts["text"].astype(str).tolist()
                )
                df_posts["sentiment"] = pd.to_numeric(df_posts["sentiment"], errors="coerce")
            else:
                df_posts["sentiment"] = 0.0

            # Erwartete Spalte: "created_utc"
            if "created_utc" in df_posts:
                df_posts["date"] = (
                    pd.to_datetime(df_posts["created_utc"], errors="coerce", utc=True)
                    .dt.tz_localize(None)
                    .dt.normalize()
                )
            else:
                df_posts["date"] = pd.NaT

            df_valid = df_posts.dropna(subset=["date"])
            if not df_valid.empty:
                sentiment_frames[ticker] = (
                    df_valid.groupby("date")["sentiment"].mean().reset_index()
                )
                series = pd.to_numeric(df_valid.get("sentiment"), errors="coerce")
                mean_val = series.mean(skipna=True)
                sentiments[ticker] = 0.0 if pd.isna(mean_val) else float(mean_val)
            else:
                sentiment_frames[ticker] = pd.DataFrame(columns=["date", "sentiment"])
                sentiments[ticker] = 0.0
        else:
            sentiment_frames[ticker] = pd.DataFrame(columns=["date", "sentiment"])
            sentiments[ticker] = 0.0

    # --- Per-Stock-Modell trainieren (parallel, read-only) ---
    train = partial(train_model_for_ticker, db_path=DB_PATH, sentiment_frames=sentiment_frames)
    with ThreadPoolExecutor() as ex:
        for t, acc, f1, roc_auc, precision, recall in ex.map(train, tickers):
            if acc is not None:
                roc_disp = roc_auc if roc_auc is not None else float("nan")
                log.info(
                    f"{t}: Modell-Accuracy {acc:.2%} | F1 {f1:.2f} | ROC-AUC {roc_disp:.2f}"
                    f" | Precision {precision:.2f} | Recall {recall:.2f}"
                )

    # Übersicht & Notify
    try:
        msg = generate_overview(tickers, reddit_posts=reddit_posts)
        notify_telegram(msg)
    except Exception as e:
        log.warning(f"Übersicht/Telegram fehlgeschlagen: {e}")

    log.info(f"🏁 Fertig in {time.time() - t0:.1f}s")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Wallenstein pipeline")
    parser.add_argument("--tickers", help="Kommagetrennte Liste von Ticker-Symbolen")
    args = parser.parse_args()
    tickers = resolve_tickers(args.tickers)
    if not tickers:
        return
    run_pipeline(tickers)


if __name__ == "__main__":
    main()
