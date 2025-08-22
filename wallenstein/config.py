import os
from dotenv import load_dotenv

# .env laden
load_dotenv()

def _require(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val

CLIENT_ID = _require("CLIENT_ID")
CLIENT_SECRET = _require("CLIENT_SECRET")
USER_AGENT = os.getenv("USER_AGENT", "Wallenstein")

GOOGLE_API_KEYFILE = os.getenv("GOOGLE_API_KEYFILE")
GOOGLE_SHEETS_ID   = os.getenv("GOOGLE_SHEETS_ID")

TELEGRAM_BOT_TOKEN = _require("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = int(os.getenv("TELEGRAM_CHAT_ID", "0"))