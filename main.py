import os
import time
import logging
from pathlib import Path
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

# Ticker
TICKERS = [t.strip().upper() for t in os.getenv("WALLENSTEIN_TICKERS", "NVDA,AMZN,SMCI").split(",") if t.strip()]

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# --- Projekt‑Module ---
from wallenstein.stock_data import update_prices, update_fx_rates
from wallenstein.db_utils import ensure_prices_view, get_latest_prices
from wallenstein.broker_targets import fetch_many
from wallenstein.db_targets import save_snapshots
from wallenstein.export_targets import export_latest_targets

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


# ---------- Main ----------
def main() -> int:
    t0 = time.time()
    log.info("🚀 Start Wallenstein: Pipeline-Run")

    # 1) Kurse und FX-Daten aktualisieren
    try:
        added = update_prices(DB_PATH, TICKERS)
        if added:
            log.info(f"✅ Kursdaten aktualisiert: +{added} neue Zeilen")
    except Exception as e:
        log.error(f"❌ Kursupdate fehlgeschlagen: {e}")

    try:
        fx_added = update_fx_rates(DB_PATH)
        log.info(f"FX-Update: +{fx_added} neue EURUSD-Zeilen")
    except Exception as e:
        log.error(f"❌ FX-Update fehlgeschlagen: {e}")

    snapshots = fetch_many(TICKERS)
    save_snapshots(DB_PATH, snapshots)
    df = export_latest_targets(DB_PATH, Path(STOCK_OVERVIEW_DIR) / "targets.csv", TICKERS)

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

    def _fmt(v: float) -> str:
        return f"{v:.0f}" if isinstance(v, (int, float)) and v == v else "n/a"

    lines = [
        f"{r.ticker}: {_fmt(r.target_mean)} USD (High {_fmt(r.target_high)} / Low {_fmt(r.target_low)}) – {r.rec_text or 'n/a'}"
        for r in df.itertuples()
    ]
    send_telegram("🎯 Analyst Targets\n" + "\n".join(lines))

    log.info(f"🏁 Fertig in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

