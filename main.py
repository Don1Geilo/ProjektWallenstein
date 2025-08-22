import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv, find_dotenv

# --- .env laden ---
env_loaded = load_dotenv(find_dotenv(usecwd=True), override=True)
if not env_loaded:
    alt_path = Path(__file__).with_name(".env")
    load_dotenv(dotenv_path=alt_path, override=True)

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("wallenstein")

# --- Pfade/Konfig ---
DB_PATH = os.getenv("WALLENSTEIN_DB_PATH", "data/wallenstein.duckdb").strip()
STOCK_OVERVIEW_DIR = "stockOverview"
os.makedirs(STOCK_OVERVIEW_DIR, exist_ok=True)

TARGETS_CSV = os.path.join(STOCK_OVERVIEW_DIR, "price_targets.csv")
RECO_CSV    = os.path.join(STOCK_OVERVIEW_DIR, "recommendations.csv")

# Ticker
TICKERS = [t.strip().upper() for t in os.getenv("WALLENSTEIN_TICKERS", "NVDA,AMZN,SMCI").split(",") if t.strip()]

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# --- Projekt‑Module ---
from wallenstein.stock_data import update_prices
from wallenstein.db_utils import ensure_prices_view, get_latest_prices_auto
from wallenstein.stock_data import update_prices, update_fx_rates
from wallenstein.db_utils import ensure_prices_view, get_latest_prices

# ---------- Utils ----------
def send_telegram(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram nicht konfiguriert – Nachricht nicht gesendet.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=15)
        if r.ok:
            try:
                data = r.json()
                mid = data.get("result", {}).get("message_id")
            except Exception:
                mid = "n/a"
            log.info(f"Telegram OK → message_id={mid}")
        else:
            log.warning(f"Telegram API {r.status_code}: {r.text[:200]}")
    except Exception as e:
        log.error(f"Telegram SendError: {e}")

def export_csv_targets(path: str, rows: List[dict]) -> None:
    cols = ["timestamp_utc","ticker","price","target_low","target_mean","target_high","rec_text","rec_mean"]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False, encoding="utf-8")

def export_csv_recommendations(path: str, rows: List[dict]) -> None:
    cols = ["timestamp_utc","ticker","price","target_low","target_mean","target_high",
            "rec_text","rec_mean","broker_sig","reddit_score","combined_score","recommendation"]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False, encoding="utf-8")

def simple_recommendation(price: Optional[float]) -> str:
    # Ohne Analysten/Reddit: neutral
    return "Hold" if price is not None else "n/a"

# ---------- Main ----------
def main() -> int:
    t0 = time.time()
    log.info("🚀 Start Wallenstein: Pipeline-Run")

    # 1) Kursdaten updaten (DB -> prices)
    try:
        added = update_prices(DB_PATH, TICKERS)
        if added:
            log.info(f"✅ Kursdaten aktualisiert: +{added} neue Zeilen")
    except Exception as e:
        log.error(f"❌ Kursupdate fehlgeschlagen: {e}")

    # 2) View aktualisieren & aktuelle Preise laden
    try:
        _ = ensure_prices_view(DB_PATH)
    except Exception as e:
        log.warning(f"View-Erstellung fehlgeschlagen/übersprungen: {e}")

    current_prices: Dict[str, Optional[float]] = {}
    try:
        current_prices = get_latest_prices_auto(DB_PATH, TICKERS)
    except Exception as e:
        log.error(f"❌ Lesen aktueller Preise fehlgeschlagen: {e}")
        current_prices = {}

    log.info(f"ℹ️ Aktuelle Preise: {current_prices}")

        # 1) Kursdaten (USD) updaten
    added = update_prices(DB_PATH, TICKERS)
    # 1b) FX EURUSD updaten
    fx_added = update_fx_rates(DB_PATH)
    log.info(f"FX-Update: +{fx_added} neue EURUSD-Zeilen")
    ensure_prices_view(DB_PATH, view_name="stocks", table_name="prices")
    prices_usd = get_latest_prices(DB_PATH, TICKERS, use_eur=False)
    prices_eur = get_latest_prices(DB_PATH, TICKERS, use_eur=True)
    log.info(f"USD: {prices_usd} | EUR: {prices_eur}")
   
    send_telegram(
    "📊 Wallenstein Preise\n" +
    "\n".join(
        f"{t}: {prices_usd.get(t):.2f} USD | {prices_eur.get(t):.2f} EUR"
        if prices_usd.get(t) is not None and prices_eur.get(t) is not None
        else f"{t}: n/a"
        for t in TICKERS
    )
)

    # 3) CSV-Exporte (ohne Analysten – Platzhalterwerte)
    now_utc = int(time.time())
    targets_rows, reco_rows = [], []

    for t in TICKERS:
        price = current_prices.get(t)
        targets_rows.append({
            "timestamp_utc": now_utc,
            "ticker": t,
            "price": price,
            "target_low": None,
            "target_mean": None,
            "target_high": None,
            "rec_text": None,
            "rec_mean": None,
        })
        reco_rows.append({
            "timestamp_utc": now_utc,
            "ticker": t,
            "price": price,
            "target_low": None,
            "target_mean": None,
            "target_high": None,
            "rec_text": None,
            "rec_mean": None,
            "broker_sig": 0.0,
            "reddit_score": 0.0,
            "combined_score": 0.0,
            "recommendation": simple_recommendation(price),
        })

    try:
        export_csv_targets(TARGETS_CSV, targets_rows)
        log.info(f"✅ price_targets.csv exportiert → {TARGETS_CSV}")
    except Exception as e:
        log.error(f"❌ Export price_targets.csv fehlgeschlagen: {e}")

    try:
        export_csv_recommendations(RECO_CSV, reco_rows)
        log.info(f"✅ recommendations.csv exportiert → {RECO_CSV}")
    except Exception as e:
        log.error(f"❌ Export recommendations.csv fehlgeschlagen: {e}")

if __name__ == "__main__":
    raise SystemExit(main())
