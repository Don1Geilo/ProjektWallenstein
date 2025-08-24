import os
from dotenv import load_dotenv

# .env laden
load_dotenv()

def _require(name: str, alt_name: str | None = None) -> str:
    """Return the value of an environment variable.

    Parameters
    ----------
    name:
        Primary environment variable name.
    alt_name:
        Optional fallback name that is checked when ``name`` is unset.
    """

    val = os.getenv(name)
    if not val and alt_name:
        val = os.getenv(alt_name)
    if not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val

# Support Reddit-specific variable names for backwards compatibility
CLIENT_ID = _require("CLIENT_ID", "REDDIT_CLIENT_ID")
CLIENT_SECRET = _require("CLIENT_SECRET", "REDDIT_CLIENT_SECRET")
USER_AGENT = os.getenv("USER_AGENT", "Wallenstein")

GOOGLE_API_KEYFILE = os.getenv("GOOGLE_API_KEYFILE")
GOOGLE_SHEETS_ID   = os.getenv("GOOGLE_SHEETS_ID")

TELEGRAM_BOT_TOKEN = _require("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = int(os.getenv("TELEGRAM_CHAT_ID", "0"))
