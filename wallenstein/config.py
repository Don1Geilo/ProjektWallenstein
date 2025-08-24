import os
from dotenv import load_dotenv

# .env laden
load_dotenv()

# Environment variables are accessed lazily so that modules only using
# Telegram notifications don't require Reddit credentials at import time.
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
USER_AGENT = os.getenv("USER_AGENT", "Wallenstein")

GOOGLE_API_KEYFILE = os.getenv("GOOGLE_API_KEYFILE")
GOOGLE_SHEETS_ID = os.getenv("GOOGLE_SHEETS_ID")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID", "0"))
