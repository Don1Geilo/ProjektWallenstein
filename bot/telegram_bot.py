"""Telegram bot command handlers using python-telegram-bot v21."""

from __future__ import annotations

import logging
from typing import List

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from wallenstein import alerts, watchlist
from wallenstein.config import settings, validate_config

log = logging.getLogger(__name__)

# Validate that required telegram credentials exist but ignore reddit ones
validate_config(require_reddit=False, require_telegram=True)

HELP_TEXT = (
    "*Wallenstein Bot*\n\n"
    "Verfügbare Befehle:\n"
    "- `/add SYMBOL` – Aktie zur Watchlist hinzufügen\n"
    "- `/remove SYMBOL` – Aktie aus Watchlist entfernen\n"
    "- `/list` – Aktuelle Watchlist anzeigen\n"
    "- `/alerts add SYMBOL OP PRICE` – Preis-Alarm setzen\n"
    "- `/alerts list` – Alle Alarme anzeigen\n"
    "- `/alerts del ID` – Alarm löschen\n"
    "- `/help` – Diese Hilfe anzeigen"
)

VALID_OPERATORS = {"<", ">", "<=", ">="}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send welcome/help text."""
    if update.message:
        await update.message.reply_text(HELP_TEXT, parse_mode="Markdown")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send help text."""
    if update.message:
        await update.message.reply_text(HELP_TEXT, parse_mode="Markdown")


async def add(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Add a ticker to the watchlist."""
    if not update.message:
        return
    if not context.args:
        await update.message.reply_text("Usage: /add SYMBOL")
        return
    symbol = context.args[0].upper()
    watchlist.add(symbol)
    await update.message.reply_text(f"Added {symbol}")


async def remove(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Remove a ticker from the watchlist."""
    if not update.message:
        return
    if not context.args:
        await update.message.reply_text("Usage: /remove SYMBOL")
        return
    symbol = context.args[0].upper()
    watchlist.remove(symbol)
    await update.message.reply_text(f"Removed {symbol}")


async def list_symbols(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List all watchlist tickers."""
    if not update.message:
        return
    symbols: List[str] = watchlist.list_symbols()
    text = "Watchlist is empty" if not symbols else "\n".join(symbols)
    await update.message.reply_text(text)


async def alerts_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle `/alerts` commands."""
    if not update.message:
        return
    if not context.args:
        await update.message.reply_text("Usage: /alerts [add|list|del]")
        return
    action = context.args[0].lower()
    if action == "add":
        if len(context.args) < 4:
            await update.message.reply_text("Usage: /alerts add SYMBOL OP PRICE")
            return
        symbol = context.args[1].upper()
        op = context.args[2]
        if op not in VALID_OPERATORS:
            await update.message.reply_text("Operator must be one of <, >, <=, >=")
            return
        try:
            price = float(context.args[3])
        except ValueError:
            await update.message.reply_text("Price must be numeric")
            return
        alert_id = alerts.add(symbol, op, price)
        await update.message.reply_text(f"Alert {alert_id} added for {symbol}")
    elif action == "list":
        all_alerts = alerts.list_alerts()
        if not all_alerts:
            await update.message.reply_text("No alerts")
            return
        lines = [f"{a['id']}: {a['symbol']} {a['op']} {a['price']}" for a in all_alerts]
        await update.message.reply_text("\n".join(lines))
    elif action in {"del", "delete"}:
        if len(context.args) < 2:
            await update.message.reply_text("Usage: /alerts del ID")
            return
        try:
            alert_id = int(context.args[1])
        except ValueError:
            await update.message.reply_text("ID must be integer")
            return
        ok = alerts.delete(alert_id)
        if ok:
            await update.message.reply_text(f"Deleted alert {alert_id}")
        else:
            await update.message.reply_text(f"Alert {alert_id} not found")
    else:
        await update.message.reply_text("Unknown subcommand")


def main() -> None:
    """Run the Telegram bot."""
    app = Application.builder().token(settings.TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("add", add))
    app.add_handler(CommandHandler("remove", remove))
    app.add_handler(CommandHandler("list", list_symbols))
    app.add_handler(CommandHandler("alerts", alerts_command))
    app.run_polling()


if __name__ == "__main__":  # pragma: no cover - manual execution only
    main()
