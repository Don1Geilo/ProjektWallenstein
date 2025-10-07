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
from wallenstein.db_utils import ensure_prices_view, get_latest_prices, upsert_predictions
from wallenstein.model_state import (
    TrainingSnapshot,
    load_training_state,
    should_skip_training,
    upsert_training_state,
)
from wallenstein.models import train_per_stock
from wallenstein.overview import OverviewMessage, generate_overview
from wallenstein.reddit_enrich import (
    compute_reddit_sentiment,
    compute_reddit_trends,
    compute_returns,
    enrich_reddit_posts,
)
from wallenstein.sentiment_analysis import analyze_sentiment_many
from wallenstein.stock_data import purge_old_prices, update_fx_rates, update_prices
from wallenstein.trending import (
    auto_add_candidates_to_watchlist,
    fetch_weekly_returns,
    scan_reddit_for_candidates,
)


AUTO_WATCHLIST_MAX_NEW = 3
AUTO_WATCHLIST_MIN_MENTIONS = 30
AUTO_WATCHLIST_MIN_LIFT = 4.0


def resolve_tickers(override: str | None = None) -> list[str]:
    """Return the list of ticker symbols to process.

    Priority:
    1. ``--tickers`` CLI override
    2. Union of ``WALLENSTEIN_TICKERS`` (env var or default) and DuckDB watchlist
    """
    # A) explizites CLI-Override
    if override:
        tickers = [t.strip().upper() for t in override.split(",") if t.strip()]
        if not tickers:
            log.warning("Keine Ticker im --tickers Override gefunden")
        return tickers

    # B) ENV-Variable (z. B. in GitHub Actions gesetzt) oder Default-Konfig
    env_tickers = (settings.WALLENSTEIN_TICKERS or "").strip()
    env_list = [t.strip().upper() for t in env_tickers.split(",") if t.strip()]

    # C) Aus DuckDB lesen (chat-übergreifend, DISTINCT)
    wl_list: list[str] = []
    try:
        from wallenstein.watchlist import all_unique_symbols as wl_all

        with duckdb.connect(DB_PATH) as con:
            try:
                added_syms = auto_add_candidates_to_watchlist(
                    con,
                    notify_fn=None,
                    max_new=AUTO_WATCHLIST_MAX_NEW,
                    min_mentions=AUTO_WATCHLIST_MIN_MENTIONS,
                    min_lift=AUTO_WATCHLIST_MIN_LIFT,
                )
                if added_syms:
                    log.info(
                        "Watchlist vor Pipeline-Lauf ergänzt: %s",
                        ", ".join(sorted(added_syms)),
                    )
            except Exception as auto_exc:  # pragma: no cover - defensive logging
                log.debug("Auto-Watchlist-Update vor Lauf fehlgeschlagen: %s", auto_exc)

            wl_list = [s.strip().upper() for s in wl_all(con)]
    except Exception as exc:  # pragma: no cover
        log.warning(f"Watchlist-Abfrage fehlgeschlagen: {exc}")

    tickers = sorted(set(env_list + wl_list))
    if not tickers:
        log.warning("Keine Ticker gefunden (ENV leer, Watchlist leer)")
    return tickers


def train_model_for_ticker(
    ticker: str,
    _con: duckdb.DuckDBPyConnection,
    price_frames: dict[str, pd.DataFrame],
    sentiment_frames: dict[str, pd.DataFrame],
) -> tuple[
    str,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    dict | None,
]:
    """Train the per-stock model for a single ticker.

    Eine offene DuckDB-Verbindung wird übergeben, wird hier jedoch nicht genutzt,
    da alle benötigten Preisdaten bereits vorgeladen sind.  ``_con`` existiert nur,
    um den Worker mit einer extern verwalteten Verbindung aufzurufen.
    """
    try:
        df_price = price_frames.get(ticker, pd.DataFrame(columns=["date", "close"])).copy()
        if df_price.empty:
            log.info(f"{ticker}: Keine Preisdaten – Training übersprungen")
            return ticker, None, None, None, None, None, None

        df_price["date"] = pd.to_datetime(df_price["date"]).dt.normalize()
        df_sent = sentiment_frames.get(ticker, pd.DataFrame(columns=["date", "sentiment"])).copy()
        if not df_sent.empty:
            df_sent["date"] = pd.to_datetime(df_sent["date"]).dt.normalize()

        df_stock = pd.merge(df_price, df_sent, on="date", how="left")
        acc, f1, roc_auc, precision, recall, meta = train_per_stock(df_stock)
        if acc is None:
            log.info(f"{ticker}: Zu wenige Daten für Modelltraining")
        return ticker, acc, f1, roc_auc, precision, recall, meta
    except Exception as e:  # pragma: no cover
        log.warning(f"{ticker}: Modelltraining fehlgeschlagen: {e}")
        return ticker, None, None, None, None, None, None


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
    if "text" not in posts:
        posts["text"] = posts["title"].fillna("") + " " + posts["selftext"].fillna("")

    sentiments = None
    if "sentiment" in posts:
        sentiments = pd.to_numeric(posts["sentiment"], errors="coerce")
        missing_mask = sentiments.isna()
        if missing_mask.any():
            new_scores = analyze_sentiment_many(
                posts.loc[missing_mask, "text"].astype(str).tolist()
            )
            sentiments.loc[missing_mask] = new_scores
        sentiments = sentiments.fillna(0.0)
    else:
        sentiments = pd.Series(
            analyze_sentiment_many(posts["text"].astype(str).tolist()), index=posts.index
        )
    posts["sentiment"] = sentiments
    posts["ups"] = pd.to_numeric(posts["ups"], errors="coerce").fillna(0).astype(int)
    posts["num_comments"] = (
        pd.to_numeric(posts["num_comments"], errors="coerce").fillna(0).astype(int)
    )
    posts["weight"] = 1 + np.log10(1 + posts["ups"]) + 0.2 * np.log10(1 + posts["num_comments"])
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


def fetch_updates(tickers: list[str]) -> dict[str, list]:
    """Fetch price, Reddit and FX updates in parallel."""
    reddit_posts: dict[str, list] = {t: [] for t in tickers}
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
    return reddit_posts


def persist_sentiment(reddit_posts: dict[str, list]) -> None:
    """Aggregate sentiment from Reddit posts and persist to DuckDB."""
    try:
        rows_list: list[dict] = []
        post_refs: list[dict] = []
        for tkr, posts in reddit_posts.items():
            for p in posts:
                post_refs.append(p)
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
        if not posts_df.empty:
            posts_df["text"] = posts_df["title"].fillna("") + " " + posts_df["selftext"].fillna("")
            sentiments = analyze_sentiment_many(posts_df["text"].astype(str).tolist())
            posts_df["sentiment"] = sentiments
            for post, score in zip(post_refs, sentiments):
                try:
                    post["sentiment"] = float(score)
                except Exception:
                    post["sentiment"] = None
        with duckdb.connect(DB_PATH) as con:
            agg = aggregate_daily_sentiment(posts_df)
            rows = upsert_reddit_daily_sentiment(con, agg)
        log.info(
            f"✅ Sentiment: {rows} rows upserted across {agg['ticker'].nunique() if not agg.empty else 0} tickers"
        )
    except Exception as e:
        log.error(f"❌ Sentiment-Aggregation fehlgeschlagen: {e}")


def _summarize_post_sentiments(posts: list[dict]) -> tuple[float | None, float | None, int, int]:
    """Return mean, median, number of valid scores and total posts for a bucket."""

    total_posts = len(posts)
    if total_posts == 0:
        return None, None, 0, 0

    scores: list[float] = []
    for post in posts:
        val = post.get("sentiment")
        if val is None:
            continue
        try:
            score = float(val)
        except (TypeError, ValueError):
            continue
        if np.isfinite(score):
            scores.append(score)

    if not scores:
        return None, None, 0, total_posts

    arr = np.asarray(scores, dtype=float)
    mean = float(np.mean(arr))
    median = float(np.median(arr))
    return mean, median, len(scores), total_posts


def _format_auto_sentiment_line(symbol: str, posts: list[dict], candidate) -> str:
    mean, median, valid, total = _summarize_post_sentiments(posts)

    if mean is None:
        detail = "keine Sentiment-Daten"
        if total:
            detail += f" ({total} Beiträge)"
    else:
        detail_parts = [f"Ø {mean:+.2f}"]
        if median is not None and not np.isclose(mean, median):
            detail_parts.append(f"Median {median:+.2f}")
        detail_parts.append(f"{valid} Beiträge")
        if total > valid:
            detail_parts.append(f"{total - valid} ohne Score")
        detail = ", ".join(detail_parts)

    extras: list[str] = []
    if candidate is not None:
        mentions = getattr(candidate, "mentions_24h", None)
        lift = getattr(candidate, "lift", None)
        if isinstance(mentions, (int, float)) and mentions > 0:
            extras.append(f"m24h={int(mentions)}")
        if isinstance(lift, (int, float)) and lift > 0:
            extras.append(f"Lift x{float(lift):.1f}")

    if extras:
        detail = detail + " | " + ", ".join(extras)

    return f"- {symbol}: {detail}"


def generate_trends(reddit_posts: dict[str, list]) -> dict[str, list]:
    """Enrich posts, compute trends, returns and scan for candidates.

    Returns a dictionary containing the trending candidates and any symbols that
    were automatically added to the watchlist so the caller can react to them
    within the same pipeline run.
    """

    known_candidates: list = []
    unknown_candidates: list = []
    added_symbols: list[str] = []

    try:
        with duckdb.connect(DB_PATH) as con:
            enriched = enrich_reddit_posts(con, reddit_posts)
            log.info(f"Reddit-Enrichment: +{enriched} Zeilen")
            trends = compute_reddit_trends(con)
            log.info(f"Trends aktualisiert: {trends}")
            returns = compute_returns(con)
            log.info(f"Returns berechnet: {returns} Posts")
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
                known_candidates = [c for c in cands if getattr(c, "is_known", True)]
                unknown_candidates = [
                    c for c in cands if not getattr(c, "is_known", True)
                ]
                if known_candidates:
                    missing_weekly = [
                        c.symbol
                        for c in known_candidates
                        if getattr(c, "weekly_return", None) is None
                    ]
                    weekly_fallback: dict[str, float] = {}
                    if missing_weekly:
                        try:
                            weekly_fallback = fetch_weekly_returns(
                                con,
                                missing_weekly,
                                max_symbols=len(missing_weekly),
                            )
                        except Exception as exc:  # pragma: no cover - best effort
                            log.debug("Weekly return lookup failed: %s", exc)
                            weekly_fallback = {}

                    def _format_candidate(cand):
                        weekly = getattr(cand, "weekly_return", None)
                        if weekly is None and weekly_fallback:
                            weekly = weekly_fallback.get(cand.symbol.upper())
                        base = f"{cand.symbol} (m24h={cand.mentions_24h}, x{cand.lift:.1f})"
                        if weekly is not None:
                            base += f", 7d {weekly * 100:+.1f}%"
                        return base

                    top_preview = ", ".join(
                        [_format_candidate(c) for c in known_candidates[:5]]
                    )
                    log.info(f"Trending-Kandidaten (Top 5, verifiziert): {top_preview}")
                    notify_telegram("🔥 Reddit-Trends: " + top_preview)
                else:
                    log.info("Trending-Kandidaten: keine verifizierten Treffer")
                if unknown_candidates:
                    unknown_preview = ", ".join(
                        [
                            f"{c.symbol} (m24h={c.mentions_24h}, x{c.lift:.1f})"
                            for c in unknown_candidates[:5]
                        ]
                    )
                    log.info(
                        "Trending-Kandidaten (unverifiziert, ignoriert): %s",
                        unknown_preview,
                    )
            else:
                log.info("Trending-Kandidaten: keine")
            added_symbols = auto_add_candidates_to_watchlist(
                con,
                notify_fn=notify_telegram,
                max_new=AUTO_WATCHLIST_MAX_NEW,
                min_mentions=AUTO_WATCHLIST_MIN_MENTIONS,
                min_lift=AUTO_WATCHLIST_MIN_LIFT,
            )
            if added_symbols:
                log.info(
                    "Auto zur Watchlist hinzugefügt: %s",
                    ", ".join(added_symbols),
                )
    except Exception as e:
        log.warning(f"Trending-Scan fehlgeschlagen: {e}")

    return {
        "known_candidates": known_candidates,
        "unknown_candidates": unknown_candidates,
        "added_symbols": added_symbols,
    }


def _build_training_snapshot(
    ticker: str,
    price_frames: dict[str, pd.DataFrame],
    sentiment_frames: dict[str, pd.DataFrame],
    reddit_posts: dict[str, list],
) -> TrainingSnapshot:
    df_price = price_frames.get(ticker)
    if df_price is None:
        df_price = pd.DataFrame(columns=["date"])
    price_dates = pd.to_datetime(df_price.get("date"), errors="coerce")
    price_dates = price_dates.dropna() if hasattr(price_dates, "dropna") else pd.Series([], dtype="datetime64[ns]")
    latest_price_date = price_dates.max().date() if not price_dates.empty else None
    price_rows = int(price_dates.size)

    df_sent = sentiment_frames.get(ticker)
    if df_sent is None:
        df_sent = pd.DataFrame(columns=["date"])
    sent_dates = pd.to_datetime(df_sent.get("date"), errors="coerce")
    sent_dates = sent_dates.dropna() if hasattr(sent_dates, "dropna") else pd.Series([], dtype="datetime64[ns]")
    latest_sentiment_date = sent_dates.max().date() if not sent_dates.empty else None
    sentiment_rows = int(sent_dates.size)

    posts = reddit_posts.get(ticker) or []
    post_timestamps = []
    for post in posts:
        created = post.get("created_utc")
        ts = pd.to_datetime(created, errors="coerce", utc=True)
        if pd.notna(ts):
            post_timestamps.append(ts)
    latest_post_utc = None
    if post_timestamps:
        latest_post = max(post_timestamps)
        if getattr(latest_post, "tzinfo", None) is not None:
            latest_post = latest_post.tz_localize(None)
        latest_post_utc = latest_post.to_pydatetime()

    return TrainingSnapshot(
        latest_price_date=latest_price_date,
        price_row_count=price_rows,
        latest_sentiment_date=latest_sentiment_date,
        sentiment_row_count=sentiment_rows,
        latest_post_utc=latest_post_utc,
    )


def train_models(tickers: list[str], reddit_posts: dict[str, list]) -> None:
    """Train per-stock models using parallel execution."""

    sentiment_frames: dict[str, pd.DataFrame] = {}
    for ticker, texts in reddit_posts.items():
        texts = list(texts) if texts is not None else []
        if texts:
            df_posts = pd.DataFrame(texts)
            if "sentiment" in df_posts:
                df_posts["sentiment"] = pd.to_numeric(
                    df_posts["sentiment"], errors="coerce"
                )
            else:
                df_posts["sentiment"] = np.nan

            missing_mask = df_posts["sentiment"].isna()
            if missing_mask.any() and "text" in df_posts:
                new_scores = analyze_sentiment_many(
                    df_posts.loc[missing_mask, "text"].astype(str).tolist()
                )
                df_posts.loc[missing_mask, "sentiment"] = new_scores

            df_posts["sentiment"] = df_posts["sentiment"].fillna(0.0)
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
            else:
                sentiment_frames[ticker] = pd.DataFrame(columns=["date", "sentiment"])
        else:
            sentiment_frames[ticker] = pd.DataFrame(columns=["date", "sentiment"])

    with duckdb.connect(DB_PATH) as con:
        placeholder = ",".join("?" * len(tickers)) or "''"
        df_prices = con.execute(
            f"SELECT ticker, date, close FROM prices WHERE ticker IN ({placeholder}) ORDER BY ticker, date",
            tickers,
        ).fetchdf()
        price_frames = {
            t: g.drop(columns="ticker") for t, g in df_prices.groupby("ticker", sort=False)
        }
        state_map = load_training_state(con)

        snapshots: dict[str, TrainingSnapshot] = {}
        trainable: list[str] = []
        skipped_same: list[str] = []
        for ticker in tickers:
            snapshot = _build_training_snapshot(ticker, price_frames, sentiment_frames, reddit_posts)
            snapshots[ticker] = snapshot
            if should_skip_training(state_map.get(ticker), snapshot):
                skipped_same.append(ticker)
                continue
            trainable.append(ticker)

        if skipped_same:
            preview = ", ".join(skipped_same[:5])
            if len(skipped_same) > 5:
                preview += ", …"
            log.info(
                "Training übersprungen für %d Ticker ohne neue Daten: %s",
                len(skipped_same),
                preview,
            )

        if not trainable:
            log.info("Keine Ticker mit neuen Daten für Modelltraining.")
            return

        train = partial(
            train_model_for_ticker,
            _con=con,
            price_frames=price_frames,
            sentiment_frames=sentiment_frames,
        )

        with ThreadPoolExecutor() as ex:
            for t, acc, f1, roc_auc, precision, recall, meta in ex.map(train, trainable):
                snapshot = snapshots.get(t)
                if acc is not None:
                    roc_disp = roc_auc if roc_auc is not None else float("nan")
                    log.info(
                        f"{t}: Modell-Accuracy {acc:.2%} | F1 {f1:.2f} | ROC-AUC {roc_disp:.2f}"
                        f" | Precision {precision:.2f} | Recall {recall:.2f}"
                    )
                avg_return_meta = meta.get("avg_strategy_return") if meta else None
                win_rate_meta = meta.get("long_win_rate") if meta else None
                upsert_training_state(
                    con,
                    t,
                    snapshot,
                    accuracy=acc,
                    f1=f1,
                    roc_auc=roc_auc,
                    precision=precision,
                    recall=recall,
                    avg_strategy_return=avg_return_meta,
                    long_win_rate=win_rate_meta,
                )

                if meta:
                    proba = meta.get("next_day_proba")
                    signal = meta.get("signal")
                    if proba is not None and signal:
                        as_of = meta.get("as_of") or datetime.now(timezone.utc)
                        horizon = int(meta.get("horizon_days", 1) or 1)
                        version = meta.get("version") or "ml-v2"
                        confidence = meta.get("confidence", proba)
                        expected_return = meta.get("expected_return")
                        backtest_return = meta.get("avg_strategy_return")
                        probability_margin = meta.get("probability_margin")
                        if expected_return is None:
                            expected_return = backtest_return
                        try:
                            written = upsert_predictions(
                                con,
                                [
                                    {
                                        "ticker": t,
                                        "as_of": as_of,
                                        "horizon_days": horizon,
                                        "signal": signal,
                                        "confidence": confidence,
                                        "expected_return": expected_return,
                                        "version": version,
                                    }
                                ],
                            )
                            if written:
                                log.info(
                                    "%s: Stored %s signal (p=%.2f, exp=%.4f, backtest=%.4f, margin=%.4f)",
                                    t,
                                    signal,
                                    proba,
                                    expected_return if expected_return is not None else float("nan"),
                                    backtest_return if backtest_return is not None else float("nan"),
                                    probability_margin if probability_margin is not None else float("nan"),
                                )
                        except Exception as exc:  # pragma: no cover - DB best effort
                            log.warning("%s: Prediction storage failed: %s", t, exc)


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

    reddit_posts = fetch_updates(tickers)
    persist_sentiment(reddit_posts)
    trend_result = generate_trends(reddit_posts)

    added_symbols = [
        str(sym).upper()
        for sym in trend_result.get("added_symbols", [])
        if isinstance(sym, str)
    ]
    candidate_map = {
        getattr(cand, "symbol", "").upper(): cand
        for cand in trend_result.get("known_candidates", [])
    }
    existing_symbols = {t.upper() for t in tickers}
    new_auto_symbols = [sym for sym in added_symbols if sym and sym not in existing_symbols]

    if new_auto_symbols:
        log.info(
            "Direktverarbeitung für neue Auto-Watchlist-Ticker: %s",
            ", ".join(new_auto_symbols),
        )
        for sym in new_auto_symbols:
            reddit_posts.setdefault(sym, [])
        try:
            added_rows = update_prices(DB_PATH, new_auto_symbols)
            if added_rows:
                log.info(
                    "📈 Nachgeladene Kursdaten für Auto-Ticker: +%d Zeilen",
                    added_rows,
                )
        except Exception as exc:  # pragma: no cover - best effort
            log.warning("Kursupdate für Auto-Ticker fehlgeschlagen: %s", exc)

        sentiment_lines = [
            _format_auto_sentiment_line(
                sym,
                reddit_posts.get(sym, []),
                candidate_map.get(sym.upper()),
            )
            for sym in new_auto_symbols
        ]
        summary_text = "\n".join(sentiment_lines).strip()
        if summary_text:
            notify_telegram("🆕 Auto-Watchlist Sentiment\n" + summary_text)

        merged = set(tickers)
        merged.update(new_auto_symbols)
        tickers = sorted(merged)

    ensure_prices_view(DB_PATH, view_name="stocks", table_name="prices")
    prices_usd = get_latest_prices(DB_PATH, tickers, use_eur=False)
    log.info(f"USD: {prices_usd}")
    if alerts_api:
        try:
            alerts_api.active_alerts(prices_usd, notify_telegram)
        except Exception as e:  # pragma: no cover
            log.warning(f"Alertprüfung fehlgeschlagen: {e}")

    train_models(tickers, reddit_posts)

    try:
        overview = generate_overview(tickers, reddit_posts=reddit_posts)
        if isinstance(overview, OverviewMessage):
            if overview.compact:
                notify_telegram(overview.compact)
            if overview.detailed:
                notify_telegram(overview.detailed)
        else:  # pragma: no cover - backward compatibility safety
            notify_telegram(str(overview))
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
