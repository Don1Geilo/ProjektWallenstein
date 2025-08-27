import logging

from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

from main import run_pipeline
from wallenstein.config import settings, validate_config
from wallenstein.overview import generate_overview

log = logging.getLogger(__name__)

validate_config()


async def handle_ticker(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Parse messages like ``!NVDA`` and reply with an overview."""
    if not update.message or not update.message.text:
        return
    text = update.message.text.strip()
    if not text.startswith("!"):
        return
    ticker = text[1:].split()[0].upper()
    log.info("Telegram request for %s", ticker)
    try:
        await context.application.run_in_executor(None, run_pipeline, [ticker])
    except Exception as exc:  # pragma: no cover - unexpected failures
        log.error("Pipeline run failed for %s: %s", ticker, exc)
        await update.message.reply_text(f"Fehler beim Aktualisieren von {ticker}: {exc}")
        return
    try:
        overview = generate_overview([ticker])
    except Exception as exc:  # pragma: no cover - network failures
        await update.message.reply_text(f"Fehler beim Abrufen von {ticker}: {exc}")
        return
    await update.message.reply_text(overview)


def main() -> None:
    """Start the Telegram bot and listen for ticker commands."""
    app = ApplicationBuilder().token(settings.TELEGRAM_BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_ticker))
    app.run_polling()


if __name__ == "__main__":
    main()
