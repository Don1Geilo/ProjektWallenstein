import os
import time
import logging
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# --- .env laden ---
env_loaded = load_dotenv(find_dotenv(usecwd=True), override=True)
if not env_loaded:
    alt_path = Path(__file__).with_name(".env")
    load_dotenv(dotenv_path=alt_path, override=True)

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("wallenstein")

try:
    from wallenstein.notify import notify_telegram
except Exception as e:  # pragma: no cover - fehlende ENV-Variablen o.Ä.
    log.warning(f"notify_telegram nicht verfügbar: {e}")

    def notify_telegram(text: str) -> bool:
        log.warning("Telegram nicht konfiguriert – Nachricht nicht gesendet.")
        return False

try:
    from wallenstein.reddit_scraper import update_reddit_data
except Exception as e:  # pragma: no cover - fehlende ENV-Variablen o.Ä.
    log.warning(f"update_reddit_data nicht verfügbar: {e}")

    def update_reddit_data(tickers, subreddits=None, limit_per_sub=50):
        return {t: [] for t in tickers}

# --- Pfade/Konfig ---
DB_PATH = os.getenv("WALLENSTEIN_DB_PATH", "data/wallenstein.duckdb").strip()
STOCK_OVERVIEW_DIR = "stockOverview"
os.makedirs(STOCK_OVERVIEW_DIR, exist_ok=True)

# Ticker
TICKERS = [t.strip().upper() for t in os.getenv("WALLENSTEIN_TICKERS", "NVDA,AMZN,SMCI").split(",") if t.strip()]

# Telegram (nicht direkt genutzt, aber zur Rückwärtskompatibilität beibehalten)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# --- Projekt‑Module ---
from wallenstein.stock_data import update_prices, update_fx_rates
from wallenstein.db_utils import ensure_prices_view, get_latest_prices
from wallenstein.sentiment import build_daily_sentiment, derive_recommendation


# ---------- Main ----------
def main() -> int:
    t0 = time.time()
    log.info("🚀 Start Wallenstein: Pipeline-Run")

    # 1) Kurse aktualisieren
    try:
        added = update_prices(DB_PATH, TICKERS)
        if added:
            log.info(f"✅ Kursdaten aktualisiert: +{added} neue Zeilen")
    except Exception as e:
        log.error(f"❌ Kursupdate fehlgeschlagen: {e}")

    # 2) Reddit-Daten aktualisieren
    try:
        reddit_posts = update_reddit_data(
            TICKERS, ["wallstreetbets", "wallstreetbetsGer", "mauerstrassenwetten"]
        )
        log.info("✅ Reddit-Daten aktualisiert")
    except Exception as e:
        log.error(f"❌ Reddit-Update fehlgeschlagen: {e}")
        reddit_posts = {t: [] for t in TICKERS}

    # 3) FX-Rates aktualisieren
    try:
        fx_added = update_fx_rates(DB_PATH)
        log.info(f"FX-Update: +{fx_added} neue EURUSD-Zeilen")
    except Exception as e:
        log.error(f"❌ FX-Update fehlgeschlagen: {e}")

    ensure_prices_view(DB_PATH, view_name="stocks", table_name="prices")
    prices_usd = get_latest_prices(DB_PATH, TICKERS, use_eur=False)
    log.info(f"USD: {prices_usd}")

    all_posts = [p for posts in reddit_posts.values() for p in posts]
    df_sentiment = build_daily_sentiment(all_posts, TICKERS)
    sentiments = {}
    for t in TICKERS:
        df_t = df_sentiment[df_sentiment["Stock"] == t]
        if not df_t.empty:
            sentiments[t] = df_t.sort_values("Date")["Sentiment"].iloc[-1]
        else:
            sentiments[t] = 0.0

    price_lines = []
    sentiment_lines = []
    for t in TICKERS:
        price = prices_usd.get(t)
        if price is not None:
            price_lines.append(f"{t}: {price:.2f} USD")
        else:
            price_lines.append(f"{t}: n/a")

        sent = sentiments.get(t, 0.0)
        rec = derive_recommendation(sent)
        sentiment_lines.append(f"{t}: Sentiment {sent:+.2f} | {rec}")

    notify_telegram(
        "📊 Wallenstein Übersicht\n" + "\n".join(price_lines + sentiment_lines)
    )

    log.info(f"🏁 Fertig in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

