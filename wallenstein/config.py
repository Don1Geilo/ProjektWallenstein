# wallenstein/config.py
import os
from dotenv import load_dotenv, find_dotenv

# .env lokal laden, CI-Secrets NICHT überschreiben
load_dotenv(find_dotenv(usecwd=True), override=False)

def _env(name: str, alt: str | None = None, default=None):
    """Hole ENV[name], optional Fallback alt, sonst default."""
    v = os.getenv(name)
    if not v and alt:
        v = os.getenv(alt)
    return v if v is not None else default

# --- Reddit (anonymer Zugriff reicht: client_id/secret/user_agent) ---
CLIENT_ID = _env("CLIENT_ID", "REDDIT_CLIENT_ID")
CLIENT_SECRET = _env("CLIENT_SECRET", "REDDIT_CLIENT_SECRET")
USER_AGENT = _env("USER_AGENT", "REDDIT_USER_AGENT", default="Wallenstein/1.0 (bot)")

def has_reddit() -> bool:
    """True, wenn anonymer Reddit-Zugriff möglich ist."""
    return bool(CLIENT_ID and CLIENT_SECRET and USER_AGENT)

# --- Google ---
GOOGLE_API_KEYFILE = _env("GOOGLE_API_KEYFILE")
GOOGLE_SHEETS_ID   = _env("GOOGLE_SHEETS_ID")

# --- Telegram (optional) ---
TELEGRAM_BOT_TOKEN = _env("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = _env("TELEGRAM_CHAT_ID")  # später zu int casten, nicht hier

def has_telegram() -> bool:
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)
