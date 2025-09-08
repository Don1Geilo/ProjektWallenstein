# wallenstein/config.py
from __future__ import annotations

import os
from pathlib import Path

# --- .env nur lokal laden (nicht im GitHub Actions CI), außer explizit erlaubt ---
if not os.getenv("GITHUB_ACTIONS") or os.getenv("ALLOW_DOTENV") == "1":
    try:
        from dotenv import find_dotenv, load_dotenv  # optional dependency

        env_path = find_dotenv(usecwd=True)
        if env_path:
            load_dotenv(env_path, override=False)  # CI-Secrets nicht überschreiben
        else:
            local_env = Path(__file__).with_name(".env")
            if local_env.exists():
                load_dotenv(local_env, override=False)
    except Exception:
        pass


def _get(*keys: str) -> str | None:
    for k in keys:
        v = os.getenv(k)
        if v is not None and str(v).strip() != "":
            return v.strip()
    return None


def _as_bool(v: str | None, default: bool = False) -> bool:
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y")


# wallenstein/config.py  (nur der relevante Ausschnitt in Settings)


class Settings:
    WALLENSTEIN_DB_PATH = _get("WALLENSTEIN_DB_PATH") or "data/wallenstein.duckdb"

    # Telegram optional
    TELEGRAM_BOT_TOKEN = _get("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = _get("TELEGRAM_CHAT_ID")

    # Reddit – akzeptiere beide Namensräume
    REDDIT_CLIENT_ID = _get("REDDIT_CLIENT_ID", "CLIENT_ID")
    REDDIT_CLIENT_SECRET = _get("REDDIT_CLIENT_SECRET", "CLIENT_SECRET")
    REDDIT_USER_AGENT = _get("REDDIT_USER_AGENT", "USER_AGENT")

    # Hugging Face Hub
    HUGGINGFACE_HUB_TOKEN = _get("HUGGINGFACE_HUB_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
    HF_HOME = _get("HF_HOME")
    TRANSFORMERS_CACHE = _get("TRANSFORMERS_CACHE")
    HF_HUB_DISABLE_TELEMETRY = _as_bool(_get("HF_HUB_DISABLE_TELEMETRY"), default=True)

    # ❗️Diese zwei Zeilen wieder hinzufügen (mit sinnvollen Defaults)
    STOOQ_USER_AGENT = _get("STOOQ_USER_AGENT") or (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    YF_USER_AGENT = _get("YF_USER_AGENT") or (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )

    USE_BERT_SENTIMENT = _as_bool(_get("USE_BERT_SENTIMENT"), default=False)
    LOG_LEVEL = _get("LOG_LEVEL") or "INFO"
    REQUEST_TIMEOUT = int(_get("REQUEST_TIMEOUT") or "15")
    MAX_RETRIES = int(_get("MAX_RETRIES") or "5")
    PIPELINE_MAX_WORKERS = int(_get("PIPELINE_MAX_WORKERS") or "4")

    WALLENSTEIN_TICKERS = _get("WALLENSTEIN_TICKERS") or "NVDA,AMZN,SMCI,TSLA,RHM.DE,NVO,UNH"
    WALLENSTEIN_DATA_SOURCE = (_get("WALLENSTEIN_DATA_SOURCE") or "stooq").lower()

    DATA_RETENTION_DAYS = int(_get("DATA_RETENTION_DAYS") or "30")
    SENTIMENT_BACKEND = (_get("SENTIMENT_BACKEND") or "finbert").lower()


settings = Settings()


# --- Exporte: setze ENV so, wie HF/Transformers es erwartet ---
def ensure_hf_env() -> None:
    # Token
    if settings.HUGGINGFACE_HUB_TOKEN and not os.getenv("HUGGINGFACE_HUB_TOKEN"):
        os.environ["HUGGINGFACE_HUB_TOKEN"] = settings.HUGGINGFACE_HUB_TOKEN
    # Caches
    if settings.HF_HOME and not os.getenv("HF_HOME"):
        os.environ["HF_HOME"] = settings.HF_HOME
    if settings.TRANSFORMERS_CACHE and not os.getenv("TRANSFORMERS_CACHE"):
        os.environ["TRANSFORMERS_CACHE"] = settings.TRANSFORMERS_CACHE
    # Telemetry off, wenn nicht explizit gesetzt
    if settings.HF_HUB_DISABLE_TELEMETRY and not os.getenv("HF_HUB_DISABLE_TELEMETRY"):
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


# Direkt beim Import sicherstellen (kannst du auch im main() aufrufen)
ensure_hf_env()

# --- Modul-Aliasse (Legacy) ---
CLIENT_ID = settings.REDDIT_CLIENT_ID
CLIENT_SECRET = settings.REDDIT_CLIENT_SECRET
USER_AGENT = settings.REDDIT_USER_AGENT
GOOGLE_API_KEYFILE = _get("GOOGLE_API_KEYFILE")
GOOGLE_SHEETS_ID = _get("GOOGLE_SHEETS_ID")
TELEGRAM_BOT_TOKEN = settings.TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID = settings.TELEGRAM_CHAT_ID


def has_reddit() -> bool:
    return bool(
        settings.REDDIT_CLIENT_ID and settings.REDDIT_CLIENT_SECRET and settings.REDDIT_USER_AGENT
    )


def has_telegram() -> bool:
    return bool(settings.TELEGRAM_BOT_TOKEN and settings.TELEGRAM_CHAT_ID)


def validate_config(require_reddit: bool = True, require_telegram: bool = False) -> None:
    missing = []
    if require_reddit:
        if not settings.REDDIT_CLIENT_ID:
            missing.append("REDDIT_CLIENT_ID/CLIENT_ID")
        if not settings.REDDIT_CLIENT_SECRET:
            missing.append("REDDIT_CLIENT_SECRET/CLIENT_SECRET")
        if not settings.REDDIT_USER_AGENT:
            missing.append("REDDIT_USER_AGENT/USER_AGENT")
    if require_telegram:
        if not settings.TELEGRAM_BOT_TOKEN:
            missing.append("TELEGRAM_BOT_TOKEN")
        if not settings.TELEGRAM_CHAT_ID:
            missing.append("TELEGRAM_CHAT_ID")
    if missing:
        raise ValueError("Missing required environment variables: " + ", ".join(missing))
