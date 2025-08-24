import logging
import os

import requests


log = logging.getLogger(__name__)


def notify_telegram(text: str) -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not text or not token or not chat_id:
        log.warning("⚠️ Telegram nicht konfiguriert oder Chat-ID fehlt.")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
        )
        ok = r.ok and r.json().get("ok", False)
        if ok:
            log.info("✅ Telegram gesendet.")
        else:
            log.warning(f"⚠️ Telegram-API Response: {r.text}")
        return ok
    except Exception as e:
        log.error(f"❌ Telegram-Error: {e}")
        print(f"❌ Telegram-Error: {e}")
        return False
