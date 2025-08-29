import logging
import os
from typing import List

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from main import run_pipeline
from wallenstein.config import settings, validate_config
from wallenstein.db import init_schema
from wallenstein.overview import generate_overview
from wallenstein.watchlist import add_ticker, list_tickers, remove_ticker

# Alerts optional (falls Modul/Funktion noch nicht vorhanden)
try:
    from wallenstein.alerts import list_alerts  # type: ignore
except Exception:  # pragma: no cover
    list_alerts = None  # type: ignore

log = logging.getLogger("wallenstein.bot")
logging.basicConfig(
    level=getattr(logging, (settings.LOG_LEVEL or "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(message)s",
)

# --- Config & DB init ---
validate_config()
DB_PATH = settings.WALLENSTEIN_DB_PATH or "wallenstein.duckdb"
init_schema(DB_PATH)  # <— WICHTIG: Pfad übergeben


HELP = (
    "📈 *Wallenstein Bot*\n"
    "• `!NVDA` – aktualisiert NVDA & zeigt Kurz-Overview\n"
    "• `/add NVDA, AMZN` – Ticker zur Watchlist hinzufügen\n"
    "• `/remove NVDA` – Ticker entfernen\n"
    "• `/list` – Watchlist anzeigen\n"
    "• `/alerts` – aktive Alerts listen (falls konfiguriert)\n"
)


# ---------- Helpers ----------
def _parse_ticker_list(args: List[str]) -> List[str]:
    raw = " ".join(args)
    parts = [p.strip().upper() for p in raw.replace(",", " ").split() if p.strip()]
    # einfache Validierung
    return [p for p in parts if p and p[0].isalnum()]


# ---------- Handlers ----------
async def handle_ticker(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Nachricht wie `!NVDA` -> Pipeline-Run + Overview senden."""
    if not update.message or not update.message.text:
        return
    text = update.message.text.strip()
    if not text.startswith("!"):
        return

    ticker = text[1:].split()[0].upper()
    log.info("Telegram request for %s", ticker)

    try:
        # Preise/Reddit/FX für EINEN Ticker ziehen
        await context.application.run_in_executor(None, run_pipeline, [ticker])
    except Exception as exc:  # pragma: no cover
        log.error("Pipeline run failed for %s: %s", ticker, exc)
        await update.message.reply_text(f"Fehler beim Aktualisieren von {ticker}: {exc}")
        return

    try:
        overview = generate_overview([ticker])
        await update.message.reply_text(overview)
    except Exception as exc:  # pragma: no cover
        await update.message.reply_text(f"Fehler beim Abrufen von {ticker}: {exc}")


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    await update.message.reply_markdown_v2(HELP)


async def cmd_add(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    if not context.args:
        await update.message.reply_text("Usage: /add TICKER[, TICKER…]")
        return

    tickers = _parse_ticker_list(context.args)
    if not tickers:
        return await update.message.reply_text("Keine gültigen Ticker erkannt.")

    for t in tickers:
        add_ticker(t)  # nutzt settings.WALLENSTEIN_DB_PATH intern
    await update.message.reply_text("Hinzugefügt: " + ", ".join(tickers))


async def cmd_remove(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    if not context.args:
        await update.message.reply_text("Usage: /remove TICKER[, TICKER…]")
        return

    tickers = _parse_ticker_list(context.args)
    if not tickers:
        return await update.message.reply_text("Keine gültigen Ticker erkannt.")

    for t in tickers:
        remove_ticker(t)
    await update.message.reply_text("Entfernt: " + ", ".join(tickers))


async def cmd_list(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    tickers = list_tickers()
    await update.message.reply_text(", ".join(tickers) if tickers else "Watchlist leer.")


async def cmd_alerts(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    if list_alerts is None:
        return await update.message.reply_text("Alerts sind (noch) nicht konfiguriert.")
    try:
        entries = list_alerts()  # falls deine Signatur anders ist, kurz anpassen
    except Exception as exc:
        return await update.message.reply_text(f"Fehler beim Lesen der Alerts: {exc}")

    if not entries:
        return await update.message.reply_text("Keine Alerts.")
    try:
        lines = [
            f"{getattr(a,'id', '?')}:{getattr(a,'ticker','?')}{getattr(a,'op','?')}{getattr(a,'price','?')} "
            f"{'on' if getattr(a,'active', False) else 'off'}"
            for a in entries
        ]
    except Exception:
        # Fallback für tuple/row Strukturen
        lines = [str(a) for a in entries]
    await update.message.reply_text("\n".join(lines))


def main() -> None:
    """Startet den Telegram-Bot und lauscht auf Befehle."""
    token = (settings.TELEGRAM_BOT_TOKEN or "").strip()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN fehlt in .env")

    log.info("Starte Telegram-Bot … (DB: %s)", os.path.abspath(DB_PATH))
    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_start))
    app.add_handler(CommandHandler("add", cmd_add))
    app.add_handler(CommandHandler("remove", cmd_remove))
    app.add_handler(CommandHandler("list", cmd_list))
    app.add_handler(CommandHandler("alerts", cmd_alerts))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_ticker))

    app.run_polling()
    log.info("Bot beendet.")


if __name__ == "__main__":
    main()
