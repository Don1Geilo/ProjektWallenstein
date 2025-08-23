import requests
from . import config

def notify_telegram(text: str) -> bool:
    if not text or not config.TELEGRAM_BOT_TOKEN or config.TELEGRAM_CHAT_ID == 0:
        print("⚠️ Telegram nicht konfiguriert oder Chat-ID fehlt.")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage",
            data={"chat_id": config.TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
        )
        ok = r.ok and r.json().get("ok", False)
        print("✅ Telegram gesendet." if ok else f"⚠️ Telegram-API Response: {r.text}")
        return ok
    except Exception as e:
        print(f"❌ Telegram-Error: {e}")
        return False
