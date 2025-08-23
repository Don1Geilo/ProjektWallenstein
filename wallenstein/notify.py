import logging

import requests
from . import config


log = logging.getLogger(__name__)

def notify_telegram(text: str) -> bool:
    if not text or not config.TELEGRAM_BOT_TOKEN or config.TELEGRAM_CHAT_ID == 0:
        log.warning("⚠️ Telegram nicht konfiguriert oder Chat-ID fehlt.")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage",
            data={"chat_id": config.TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
        )
        ok = r.ok and r.json().get("ok", False)
        if ok:
            log.info("✅ Telegram gesendet.")
        else:
            log.warning(f"⚠️ Telegram-API Response: {r.text}")
        return ok
    except Exception as e:
        log.error(f"❌ Telegram-Error: {e}")
        return False
