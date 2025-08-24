import os
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv, find_dotenv
import pandas as pd
import duckdb

# --- .env laden ---
env_loaded = load_dotenv(find_dotenv(usecwd=True), override=True)
if not env_loaded:
    alt_path = Path(__file__).with_name(".env")
    load_dotenv(dotenv_path=alt_path, override=True)

# --- Logging ---
LOG_LEVEL = os.getenv("WALLENSTEIN_LOG_LEVEL", "INFO").upper()
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

# --- Pfade/Konfig ---
DB_PATH = os.getenv("WALLENSTEIN_DB_PATH", "data/wallenstein.duckdb").strip()
STOCK_OVERVIEW_DIR = "stockOverview"
os.makedirs(STOCK_OVERVIEW_DIR, exist_ok=True)
os.makedirs(Path(DB_PATH).parent, exist_ok=True)  # stellt sicher, dass data/ existiert

# Ticker (Standard inkl. TSLA)
TICKERS = [
    t.strip().upper()
    for t in os.getenv("WALLENSTEIN_TICKERS", "NVDA,AMZN,SMCI,TSLA").split(",")
    if t.strip()
]

# Telegram (zur Rückwärtskompatibilität beibehalten)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# --- Projekt-Module ---
from wallenstein.stock_data import update_prices, update_fx_rates
from wallenstein.db_utils import ensure_prices_view, get_latest_prices
from wallenstein.sentiment import analyze_sentiment_batch
from wallenstein.overview import generate_overview
from wallenstein.models import train_per_stock


# ---------- Main ----------
def run_pipeline(tickers: list[str] | None = None) -> int:
    t0 = time.time()
    log.info("🚀 Start Wallenstein: Pipeline-Run")

    if tickers is None:
        tickers = TICKERS

    reddit_posts = {t: [] for t in tickers}
    added = 0
    fx_added = 0

    # Parallel: Preise, Reddit, FX
    with ThreadPoolExecutor() as executor:
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

    # View sicherstellen & Preise ziehen
    ensure_prices_view(DB_PATH, view_name="stocks", table_name="prices")
    prices_usd = get_latest_prices(DB_PATH, tickers, use_eur=False)
    log.info(f"USD: {prices_usd}")

    # Sentiment je Ticker aus Reddit-Posts
    sentiments: dict[str, float] = {}
    sentiment_frames: dict[str, pd.DataFrame] = {}

    for ticker, texts in reddit_posts.items():
        texts = list(texts) if texts is not None else []
        if texts:
            df_posts = pd.DataFrame(texts)

            # Erwartete Spalte: "text" (ansonsten leere Sentiments)
            if "text" in df_posts:
                df_posts["sentiment"] = analyze_sentiment_batch(
                    df_posts["text"].astype(str).tolist()
                )
            else:
                df_posts["sentiment"] = 0.0

            # Erwartete Spalte: "created_utc" (Unix oder ISO)
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
                sentiments[ticker] = float(df_valid["sentiment"].mean())
            else:
                sentiment_frames[ticker] = pd.DataFrame(columns=["date", "sentiment"])
                sentiments[ticker] = 0.0
        else:
            sentiments[ticker] = 0.0
            sentiment_frames[ticker] = pd.DataFrame(columns=["date", "sentiment"])

    # --- Per-Stock-Modell trainieren (parallel, read-only Zugriffe auf DuckDB sind ok) ---
    def _train(t: str):
        try:
            with duckdb.connect(DB_PATH) as con:
                df_price = con.execute(
                    "SELECT date, close FROM prices WHERE ticker = ? ORDER BY date",
                    [t],
                ).fetchdf()
            if df_price.empty:
                log.info(f"{t}: Keine Preisdaten – Training übersprungen")
                return t, None, None

            df_price["date"] = pd.to_datetime(df_price["date"]).dt.normalize()
            df_sent = sentiment_frames.get(
                t, pd.DataFrame(columns=["date", "sentiment"])
            ).copy()
            if not df_sent.empty:
                df_sent["date"] = pd.to_datetime(df_sent["date"]).dt.normalize()

            df_stock = pd.merge(df_price, df_sent, on="date", how="left")
            acc, f1 = train_per_stock(df_stock)
            if acc is None:
                log.info(f"{t}: Zu wenige Daten für Modelltraining")
            return t, acc, f1
        except Exception as e:
            log.warning(f"{t}: Modelltraining fehlgeschlagen: {e}")
            return t, None, None

    with ThreadPoolExecutor() as ex:
        for t, acc, f1 in ex.map(_train, tickers):
            if acc is not None:
                log.info(f"{t}: Modell-Accuracy {acc:.2%} | F1 {f1:.2f}")

    # Übersicht & Notify
    try:
        msg = generate_overview(tickers)
        notify_telegram(msg)
    except Exception as e:
        log.warning(f"Übersicht/Telegram fehlgeschlagen: {e}")

    log.info(f"🏁 Fertig in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    run_pipeline()
