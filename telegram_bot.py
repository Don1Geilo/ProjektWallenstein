import logging

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from main import run_pipeline
from wallenstein.alerts import list_alerts
from wallenstein.config import settings, validate_config
from wallenstein.db import init_schema
from wallenstein.overview import generate_overview
from wallenstein.watchlist import add_ticker, list_tickers, remove_ticker

log = logging.getLogger(__name__)

validate_config()
init_schema()


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


async def cmd_add(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    if not context.args:
        await update.message.reply_text("Usage: /add TICKER")
        return
    ticker = context.args[0].upper()
    add_ticker(ticker)
    await update.message.reply_text(f"{ticker} added.")


async def cmd_remove(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    if not context.args:
        await update.message.reply_text("Usage: /remove TICKER")
        return
    ticker = context.args[0].upper()
    remove_ticker(ticker)
    await update.message.reply_text(f"{ticker} removed.")


async def cmd_list(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    tickers = list_tickers()
    if tickers:
        await update.message.reply_text(", ".join(tickers))
    else:
        await update.message.reply_text("Watchlist empty.")


async def cmd_alerts(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    alerts = list_alerts()
    if not alerts:
        await update.message.reply_text("No alerts.")
        return
    lines = [f"{a.id}:{a.ticker}{a.op}{a.price} {'on' if a.active else 'off'}" for a in alerts]
    await update.message.reply_text("\n".join(lines))


def main() -> None:
    """Start the Telegram bot and listen for ticker and watchlist commands."""
    app = ApplicationBuilder().token(settings.TELEGRAM_BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_ticker))
    app.add_handler(CommandHandler("add", cmd_add))
    app.add_handler(CommandHandler("remove", cmd_remove))
    app.add_handler(CommandHandler("list", cmd_list))
    app.add_handler(CommandHandler("alerts", cmd_alerts))
    app.run_polling()


if __name__ == "__main__":
    main()
