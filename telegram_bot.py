import logging

from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

from wallenstein.reddit_scraper import update_reddit_data
from wallenstein.notify import notify_telegram
from wallenstein import config

log = logging.getLogger(__name__)

async def handle_ticker(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Parse messages like ``!NVDA`` and update Reddit data for the ticker."""
    if not update.message or not update.message.text:
        return
    text = update.message.text.strip()
    if not text.startswith("!"):
        return
    ticker = text[1:].split()[0].upper()
    log.info("Telegram request for %s", ticker)
    try:
        data = update_reddit_data([ticker])
    except Exception as exc:  # pragma: no cover - network failures
        await update.message.reply_text(f"Fehler beim Abrufen von {ticker}: {exc}")
        return
    posts = data.get(ticker, [])
    msg = f"{ticker}: {len(posts)} Reddit posts gefunden."
    await update.message.reply_text(msg)
    try:
        notify_telegram(msg)
    except Exception:
        # ignore optional broadcast errors
        pass

def main() -> None:
    """Start the Telegram bot and listen for ticker commands."""
    app = ApplicationBuilder().token(config.TELEGRAM_BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_ticker))
    app.run_polling()


if __name__ == "__main__":
    main()
